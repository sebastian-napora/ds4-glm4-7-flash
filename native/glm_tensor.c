#include "glm_tensor.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_err[512];
const char *glm_tensor_error(void) { return g_err; }

static void set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
}

static const glm_gguf_tensor *find_tensor(const glm_gguf_model *m, const char *name) {
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    }
    return NULL;
}

static inline float half_to_float(uint16_t h);

// ─── Q8_0 row dequantization ───────────────────────────────────────────────────
// Q8_0: 34 bytes per block of 32 values.
// Layout per block (matches llama.cpp `block_q8_0`):
//   d  : fp16 scale  (2 bytes at offset 0)
//   qs : int8[32]    (32 bytes at offset 2)
static inline void dequant_row_q8(const int8_t *src, float *dst, uint32_t n_cols) {
    uint32_t blk = 0;
    while (blk * 32 < n_cols) {
        const uint8_t *u = (const uint8_t *)src;
        uint16_t d_raw = (uint16_t)u[0] | ((uint16_t)u[1] << 8);
        float scale = half_to_float(d_raw);
        const int8_t *qs = src + 2;
        for (uint32_t i = 0; i < 32 && (blk * 32 + i) < n_cols; i++) {
            dst[blk * 32 + i] = (float)qs[i] * scale;
        }
        src += 34;
        blk++;
    }
}

// ─── half to float ─────────────────────────────────────────────────────────────
static inline float half_to_float(uint16_t h) {
    unsigned int sign = h >> 15;
    int exp = (h >> 10) & 0x1f;
    unsigned int frac = h & 0x3ff;
    if (exp == 0) {
        // Subnormal or zero
        if (frac == 0) return sign ? -0.0f : 0.0f;
        // Subnormal: value = (-1)^sign * 2^(-14) * (frac / 1024)
        return sign ? -(float)frac / 1048576.0f : (float)frac / 1048576.0f;
    }
    if (exp == 31) {
        return sign ? -1.0f / 0.0f : 1.0f / 0.0f;
    }
    unsigned int bits = (sign << 31) | ((unsigned int)(exp - 15 + 127) << 23) | (frac << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

// ─── Q6_K row dequantization ──────────────────────────────────────────────────
// Q6_K (GGUF type 14): 210 bytes per 256-element superblock (QK_K=256).
// Layout per superblock:
//   ql[128]  : lower 4 bits packed 2-per-byte (256 nibbles total)
//   qh[64]   : upper 2 bits packed 4-per-byte (256 2-bit values)
//   scales[16]: one int8 scale per group of 16 values
//   d        : fp16 super-block scale (2 bytes, at offset 208)
// Mirrors llama.cpp's `dequantize_row_q6_K` (the canonical 128-stride
// unpacking pattern). Reconstruction per value: q = ((ql_nibble) | (qh_2bits << 4)) - 32.
static inline void dequant_row_q6(const uint8_t *src, float *dst, uint32_t n_cols) {
    uint32_t total = (n_cols + 255) / 256;
    for (uint32_t sblk = 0; sblk < total; sblk++) {
        // Per-superblock pointers — MUST be recomputed each iteration.
        const uint8_t *ql = src;
        const uint8_t *qh = src + 128;
        const int8_t  *sc = (const int8_t *)(src + 192);
        const float    d  = half_to_float(((const uint16_t *)(src + 208))[0]);

        uint32_t base = sblk * 256;
        // Each superblock has 2 halves of 128 values; within each half, 4 quartets of 32 values.
        // n iterates over halves (0 and 128); l iterates within a half (0..31).
        for (uint32_t n = 0; n < 256; n += 128) {
            for (uint32_t l = 0; l < 32; l++) {
                int is = (int)(l / 16);
                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                uint32_t v1 = base + n + l +  0;
                uint32_t v2 = base + n + l + 32;
                uint32_t v3 = base + n + l + 64;
                uint32_t v4 = base + n + l + 96;
                if (v1 < n_cols) dst[v1] = d * (float)sc[is + 0] * (float)q1;
                if (v2 < n_cols) dst[v2] = d * (float)sc[is + 2] * (float)q2;
                if (v3 < n_cols) dst[v3] = d * (float)sc[is + 4] * (float)q3;
                if (v4 < n_cols) dst[v4] = d * (float)sc[is + 6] * (float)q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }

        src += 210;
    }
}

static bool read_f32(const glm_gguf_model *m, const glm_gguf_tensor *t,
                      float *dst, uint64_t off, uint64_t *inout_fpos) {
    (void)off; // tensor offset is always relative to the tensor-data section base
    uint64_t elems = glm_gguf_tensor_elems(t);
    if (fseek(m->fp, (long)(m->tensor_data_offset + t->offset), SEEK_SET) != 0) return false;
    if (t->type == T_F32) {
        if (fread(dst, 1, (size_t)(elems * 4), m->fp) != (size_t)(elems * 4)) return false;
    } else {
        uint16_t *buf = (uint16_t *)malloc((size_t)(elems * 2));
        if (!buf) return false;
        if (fread(buf, 1, (size_t)(elems * 2), m->fp) != (size_t)(elems * 2)) {
            free(buf); return false;
        }
        for (uint64_t i = 0; i < elems; i++) {
            dst[i] = half_to_float(buf[i]);
        }
        free(buf);
    }
    if (inout_fpos) *inout_fpos = (uint64_t)ftell(m->fp);
    return true;
}

static bool load_f32(const glm_gguf_model *m, const glm_gguf_tensor *t,
                     glm_tensor_f32 *out, uint64_t off, uint64_t *inout_fpos) {
    (void)off;
    uint64_t rows = t->dim[0];
    uint64_t cols = t->ndim >= 2 ? t->dim[1] : 1;
    out->data = (float *)calloc(rows * cols, sizeof(float));
    if (!out->data) { set_err("OOM"); return false; }
    out->rows = (uint32_t)rows;
    out->cols = (uint32_t)cols;
    out->is_quant = false;
    if (!read_f32(m, t, out->data, off, inout_fpos)) { free(out->data); return false; }
    return true;
}

// ── Generic 2-D quantized tensor loader (Q8_0 or Q6_K) ─────────────────────
// Handles tensors with potentially unreliable offset metadata by falling back
// to current file position when stored offset is invalid.
static bool load_quant_2d(const glm_gguf_model *m, const glm_gguf_tensor *t,
                         glm_tensor_f32 *out, uint64_t base_off, uint64_t *inout_fpos) {
    (void)base_off;
    if (t->ndim != 2) {
        set_err("tensor %s has %u dims, expected 2", t->name, t->ndim);
        return false;
    }
    uint64_t rows = t->dim[0];
    uint64_t cols = t->dim[1];

    if (t->type == T_Q8_0) {
        uint32_t blk_per_row = (uint32_t)((cols + 31) / 32);
        uint64_t row_bytes = (uint64_t)blk_per_row * 34;
        out->data = (float *)calloc(rows * cols, sizeof(float));
        if (!out->data) { set_err("OOM %s rows=%lu cols=%lu", t->name, (unsigned long)rows, (unsigned long)cols); return false; }
        out->rows = (uint32_t)rows; out->cols = (uint32_t)cols; out->is_quant = false;
        uint64_t seek_pos = m->tensor_data_offset + t->offset;
        if (seek_pos >= (uint64_t)1 << 62) { set_err("%s: implausible offset %lu", t->name, (unsigned long)seek_pos); return false; }
        if (fseek(m->fp, (long)seek_pos, SEEK_SET) != 0) { set_err("%s seek %lu failed", t->name, (unsigned long)seek_pos); free(out->data); return false; }
        uint8_t *buf = (uint8_t *)malloc((size_t)row_bytes);
        if (!buf) { set_err("%s buf alloc failed", t->name); free(out->data); return false; }
        for (uint64_t r = 0; r < rows; r++) {
            if (fread(buf, 1, (size_t)row_bytes, m->fp) != row_bytes) {
                set_err("%s short read row %lu/%lu seek=%lu rb=%lu", t->name, (unsigned long)r, (unsigned long)rows, (unsigned long)seek_pos, (unsigned long)row_bytes);
                free(buf); free(out->data); return false;
            }
            dequant_row_q8((const int8_t *)buf, out->data + r * cols, (uint32_t)cols);
        }
        free(buf);
        if (inout_fpos) *inout_fpos = (uint64_t)ftell(m->fp);
        return true;
    } else if (t->type == T_Q6_K) {
        uint32_t sup_per_row = (uint32_t)((cols + 255) / 256);
        uint64_t row_bytes = (uint64_t)sup_per_row * 210;
        out->data = (float *)calloc(rows * cols, sizeof(float));
        if (!out->data) { set_err("OOM"); return false; }
        out->rows = (uint32_t)rows; out->cols = (uint32_t)cols; out->is_quant = false;
        uint64_t seek_pos = m->tensor_data_offset + t->offset;
        if (seek_pos >= (uint64_t)1 << 62) { set_err("%s: implausible offset %lu", t->name, (unsigned long)seek_pos); return false; }
        if (fseek(m->fp, (long)seek_pos, SEEK_SET) != 0) { free(out->data); return false; }
        uint8_t *buf = (uint8_t *)malloc((size_t)row_bytes);
        if (!buf) { free(out->data); return false; }
        for (uint64_t r = 0; r < rows; r++) {
            if (fread(buf, 1, (size_t)row_bytes, m->fp) != row_bytes) {
                free(buf); free(out->data); return false;
            }
            dequant_row_q6(buf, out->data + r * cols, (uint32_t)cols);
        }
        free(buf);
        if (inout_fpos) *inout_fpos = (uint64_t)ftell(m->fp);
        return true;
    } else {
        free(out->data);
        set_err("tensor %s has unsupported quant type %u", t->name, t->type);
        return false;
    }
}

// attn_k_b [192][512][20] -> [20*192=3840][512]
// attn_v_b [512][256][20] -> [20*512=10240][256]
// GGUF layout: [inner][mid][heads]. Our dest: [heads*inner][mid]
static bool load_q8_3d_attn(const glm_gguf_model *m, const glm_gguf_tensor *t,
                              glm_tensor_f32 *out, uint64_t off, uint64_t *inout_fpos) {
    (void)off;
    if (t->ndim != 3) { set_err("tensor %s is not 3D", t->name); return false; }
    uint64_t inner = t->dim[0];
    uint64_t mid   = t->dim[1];
    uint64_t heads = t->dim[2];
    uint64_t rows = inner * heads;
    uint64_t cols = mid;

    out->data = (float *)calloc(rows * cols, sizeof(float));
    if (!out->data) { set_err("OOM"); return false; }
    out->rows = (uint32_t)rows; out->cols = (uint32_t)cols; out->is_quant = false;

    uint64_t seek_pos = m->tensor_data_offset + t->offset;
    if (seek_pos >= (uint64_t)1 << 62) { set_err("%s bad offset", t->name); return false; }
    if (fseek(m->fp, (long)seek_pos, SEEK_SET) != 0) { free(out->data); return false; }

    if (t->type == T_Q8_0) {
        uint32_t blk_per_row = (uint32_t)((cols + 31) / 32);
        uint64_t row_bytes = (uint64_t)blk_per_row * 34;
        uint8_t *buf = (uint8_t *)malloc((size_t)row_bytes);
        if (!buf) { free(out->data); return false; }
        for (uint64_t h = 0; h < heads; h++) {
            for (uint64_t i = 0; i < inner; i++) {
                if (fread(buf, 1, (size_t)row_bytes, m->fp) != row_bytes) {
                    free(buf); free(out->data); return false;
                }
                dequant_row_q8((const int8_t *)buf, out->data + (h * inner + i) * cols, (uint32_t)cols);
            }
        }
        free(buf);
    } else if (t->type == T_Q6_K) {
        uint32_t sup_per_row = (uint32_t)((cols + 255) / 256);
        uint64_t row_bytes = (uint64_t)sup_per_row * 210;
        uint8_t *buf = (uint8_t *)malloc((size_t)row_bytes);
        if (!buf) { free(out->data); return false; }
        for (uint64_t h = 0; h < heads; h++) {
            for (uint64_t i = 0; i < inner; i++) {
                if (fread(buf, 1, (size_t)row_bytes, m->fp) != row_bytes) {
                    free(buf); free(out->data); return false;
                }
                dequant_row_q6(buf, out->data + (h * inner + i) * cols, (uint32_t)cols);
            }
        }
        free(buf);
    } else {
        free(out->data);
        set_err("tensor %s has unsupported type %u for 3D attn", t->name, t->type);
        return false;
    }
    if (inout_fpos) *inout_fpos = (uint64_t)ftell(m->fp);
    return true;
}

// ffn_gate_exps [2048][1536][64] -> [131072][1536]
// ffn_up_exps [2048][1536][64] -> [131072][1536]
// ffn_down_exps [1536][2048][64] -> [98304][2048]
// GGUF: [F][H][E] row-major. Dest: [E*F][H] rows interleaved by expert.
static bool load_q8_3d_moe(const glm_gguf_model *m, const glm_gguf_tensor *t,
                            glm_tensor_f32 *out, uint64_t off, uint64_t *inout_fpos) {
    (void)off;
    if (t->ndim != 3) { set_err("tensor %s is not 3D", t->name); return false; }
    uint64_t F = t->dim[0];
    uint64_t H = t->dim[1];
    uint64_t E = t->dim[2];
    uint64_t rows = F * E;
    uint64_t cols = H;

    out->data = (float *)calloc(rows * cols, sizeof(float));
    if (!out->data) { set_err("OOM"); return false; }
    out->rows = (uint32_t)rows; out->cols = (uint32_t)cols; out->is_quant = false;

    uint64_t seek_pos = m->tensor_data_offset + t->offset;
    if (seek_pos >= (uint64_t)1 << 62) { set_err("%s bad offset", t->name); return false; }
    if (fseek(m->fp, (long)seek_pos, SEEK_SET) != 0) { free(out->data); return false; }

    if (t->type == T_Q8_0) {
        uint32_t blk_per_row = (uint32_t)((cols + 31) / 32);
        uint64_t row_bytes = (uint64_t)blk_per_row * 34;
        uint8_t *buf = (uint8_t *)malloc((size_t)row_bytes);
        if (!buf) { free(out->data); return false; }
        for (uint64_t e = 0; e < E; e++) {
            for (uint64_t f = 0; f < F; f++) {
                if (fread(buf, 1, (size_t)row_bytes, m->fp) != row_bytes) {
                    free(buf); free(out->data); return false;
                }
                dequant_row_q8((const int8_t *)buf, out->data + (e * F + f) * cols, (uint32_t)cols);
            }
        }
        free(buf);
    } else if (t->type == T_Q6_K) {
        uint32_t sup_per_row = (uint32_t)((cols + 255) / 256);
        uint64_t row_bytes = (uint64_t)sup_per_row * 210;
        uint8_t *buf = (uint8_t *)malloc((size_t)row_bytes);
        if (!buf) { free(out->data); return false; }
        for (uint64_t e = 0; e < E; e++) {
            for (uint64_t f = 0; f < F; f++) {
                if (fread(buf, 1, (size_t)row_bytes, m->fp) != row_bytes) {
                    free(buf); free(out->data); return false;
                }
                dequant_row_q6(buf, out->data + (e * F + f) * cols, (uint32_t)cols);
            }
        }
        free(buf);
    } else {
        free(out->data);
        set_err("tensor %s has unsupported type %u for 3D MoE", t->name, t->type);
        return false;
    }
    if (inout_fpos) *inout_fpos = (uint64_t)ftell(m->fp);
    return true;
}

void glm_tensor_f32_free(glm_tensor_f32 *t) {
    if (!t) return;
    free(t->data);
    t->data = NULL;
}

static bool load_layer(const glm_gguf_model *m, struct layer_weights *lw,
                       int layer, uint64_t off) {
    const glm_gguf_tensor *t;
    char key[128];

    // Attention (all layers)
    snprintf(key, sizeof(key), "blk.%d.attn_q_a.weight", layer);
    t = find_tensor(m, key);
    if (!t || !load_quant_2d(m, t, &lw->attn_q_a, off, &off)) { if (!t) set_err("missing %s", key); return false; }

    snprintf(key, sizeof(key), "blk.%d.attn_q_a_norm.weight", layer);
    t = find_tensor(m, key);
    if (!t || !load_f32(m, t, &lw->attn_q_a_norm, off, &off)) { if (!t) set_err("missing %s", key); return false; }

    snprintf(key, sizeof(key), "blk.%d.attn_q_b.weight", layer);
    t = find_tensor(m, key);
    if (!t || !load_quant_2d(m, t, &lw->attn_q_b, off, &off)) { if (!t) set_err("missing %s", key); return false; }

    snprintf(key, sizeof(key), "blk.%d.attn_k_b.weight", layer);
    t = find_tensor(m, key);
    if (!t || t->ndim != 3 || !load_q8_3d_attn(m, t, &lw->attn_k_b, off, &off)) { if (!t) set_err("missing %s", key); return false; }

    snprintf(key, sizeof(key), "blk.%d.attn_kv_a_mqa.weight", layer);
    t = find_tensor(m, key);
    if (!t || !load_quant_2d(m, t, &lw->attn_kv_a_mqa, off, &off)) { if (!t) set_err("missing %s", key); return false; }

    snprintf(key, sizeof(key), "blk.%d.attn_kv_a_norm.weight", layer);
    t = find_tensor(m, key);
    if (!t || !load_f32(m, t, &lw->attn_kv_a_norm, off, &off)) { if (!t) set_err("missing %s", key); return false; }

    snprintf(key, sizeof(key), "blk.%d.attn_v_b.weight", layer);
    t = find_tensor(m, key);
    if (!t || t->ndim != 3 || !load_q8_3d_attn(m, t, &lw->attn_v_b, off, &off)) { if (!t) set_err("missing %s", key); return false; }

    snprintf(key, sizeof(key), "blk.%d.attn_output.weight", layer);
    t = find_tensor(m, key);
    if (!t || !load_quant_2d(m, t, &lw->attn_output, off, &off)) { if (!t) set_err("missing %s", key); return false; }

    snprintf(key, sizeof(key), "blk.%d.attn_norm.weight", layer);
    t = find_tensor(m, key);
    if (!t || !load_f32(m, t, &lw->attn_norm, off, &off)) { if (!t) set_err("missing %s", key); return false; }

    if (layer == 0) {
        snprintf(key, sizeof(key), "blk.%d.ffn_gate.weight", layer);
        t = find_tensor(m, key);
        if (!t || !load_quant_2d(m, t, &lw->ffn_gate, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.ffn_up.weight", layer);
        t = find_tensor(m, key);
        if (!t || !load_quant_2d(m, t, &lw->ffn_up, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.ffn_down.weight", layer);
        t = find_tensor(m, key);
        if (!t || !load_quant_2d(m, t, &lw->ffn_down, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.ffn_norm.weight", layer);
        t = find_tensor(m, key);
        if (!t || !load_f32(m, t, &lw->ffn_norm, off, &off)) { if (!t) set_err("missing %s", key); return false; }
    } else {
        snprintf(key, sizeof(key), "blk.%d.ffn_gate_inp.weight", layer);
        t = find_tensor(m, key);
        if (!t) { if (!t) set_err("missing %s", key); return false; }
        if (!load_f32(m, t, &lw->ffn_gate_inp, off, &off)) return false;

        snprintf(key, sizeof(key), "blk.%d.ffn_gate_exps.weight", layer);
        t = find_tensor(m, key);
        if (!t || t->ndim != 3 || !load_q8_3d_moe(m, t, &lw->ffn_gate_exps, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.ffn_up_exps.weight", layer);
        t = find_tensor(m, key);
        if (!t || t->ndim != 3 || !load_q8_3d_moe(m, t, &lw->ffn_up_exps, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.ffn_down_exps.weight", layer);
        t = find_tensor(m, key);
        if (!t || t->ndim != 3 || !load_q8_3d_moe(m, t, &lw->ffn_down_exps, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.ffn_gate_shexp.weight", layer);
        t = find_tensor(m, key);
        if (!t || !load_quant_2d(m, t, &lw->ffn_gate_shexp, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.ffn_up_shexp.weight", layer);
        t = find_tensor(m, key);
        if (!t || !load_quant_2d(m, t, &lw->ffn_up_shexp, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.ffn_down_shexp.weight", layer);
        t = find_tensor(m, key);
        if (!t || !load_quant_2d(m, t, &lw->ffn_down_shexp, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.exp_probs_b.bias", layer);
        t = find_tensor(m, key);
        if (!t || !load_f32(m, t, &lw->exp_probs_b, off, &off)) { if (!t) set_err("missing %s", key); return false; }

        snprintf(key, sizeof(key), "blk.%d.ffn_norm.weight", layer);
        t = find_tensor(m, key);
        if (!t || !load_f32(m, t, &lw->ffn_norm_moe, off, &off)) { if (!t) set_err("missing %s", key); return false; }
    }

    return true;
}

bool glm_weights_load(glm_weights *w, const glm_gguf_model *m) {
    const glm_gguf_tensor *t;
    uint64_t base_off = m->tensor_data_offset;
    memset(w, 0, sizeof(*w));

    fprintf(stderr, "Loading model weights...\n");

    t = find_tensor(m, "token_embd.weight");
    if (!t) { set_err("missing token_embd.weight"); return false; }
    fprintf(stderr, "  token_embd: %lu x %lu (type %u) t->offset=%lu base_off=%lu\n",
            (unsigned long)t->dim[0], (unsigned long)t->dim[1], t->type,
            (unsigned long)t->offset, (unsigned long)base_off);
    uint64_t off = (t->offset > 0) ? (base_off + t->offset) : base_off;
    if (!load_quant_2d(m, t, &w->token_embd, base_off, &off)) return false;

    t = find_tensor(m, "output.weight");
    if (!t) { set_err("missing output.weight"); return false; }
    fprintf(stderr, "  output: %lu x %lu (type %u) t->offset=%lu\n",
            (unsigned long)t->dim[0], (unsigned long)t->dim[1], t->type, (unsigned long)t->offset);
    off = (t->offset > 0) ? (base_off + t->offset) : base_off;
    if (!load_quant_2d(m, t, &w->output, base_off, &off)) return false;

    t = find_tensor(m, "output_norm.weight");
    if (!t) { set_err("missing output_norm.weight"); return false; }
    if (!load_f32(m, t, &w->output_norm, off, &off)) return false;

    fprintf(stderr, "  global: %.1f MB\n",
            (float)(((uint64_t)w->token_embd.rows * w->token_embd.cols +
                     (uint64_t)w->output.rows * w->output.cols +
                     (uint64_t)w->output_norm.rows) * 4ULL) / (1024.0f * 1024.0f));

    for (int l = 0; l < GLM_LAYERS; l++) {
        fprintf(stderr, "  layer %2d...", l);
        fflush(stderr);
        if (!load_layer(m, &w->layers[l], l, off)) return false;
        fprintf(stderr, " OK\n");
    }

    uint64_t total_f32 = 0;
    total_f32 += (uint64_t)w->token_embd.rows * w->token_embd.cols;
    total_f32 += (uint64_t)w->output.rows * w->output.cols;
    total_f32 += (uint64_t)w->output_norm.rows;
    for (int l = 0; l < GLM_LAYERS; l++) {
        total_f32 += (uint64_t)w->layers[l].attn_q_a.rows * w->layers[l].attn_q_a.cols;
        total_f32 += w->layers[l].attn_q_a_norm.rows;
        total_f32 += (uint64_t)w->layers[l].attn_q_b.rows * w->layers[l].attn_q_b.cols;
        total_f32 += (uint64_t)w->layers[l].attn_k_b.rows * w->layers[l].attn_k_b.cols;
        total_f32 += (uint64_t)w->layers[l].attn_kv_a_mqa.rows * w->layers[l].attn_kv_a_mqa.cols;
        total_f32 += w->layers[l].attn_kv_a_norm.rows;
        total_f32 += (uint64_t)w->layers[l].attn_v_b.rows * w->layers[l].attn_v_b.cols;
        total_f32 += (uint64_t)w->layers[l].attn_output.rows * w->layers[l].attn_output.cols;
        total_f32 += w->layers[l].attn_norm.rows;
        if (l == 0) {
            total_f32 += (uint64_t)w->layers[l].ffn_gate.rows * w->layers[l].ffn_gate.cols;
            total_f32 += (uint64_t)w->layers[l].ffn_up.rows * w->layers[l].ffn_up.cols;
            total_f32 += (uint64_t)w->layers[l].ffn_down.rows * w->layers[l].ffn_down.cols;
            total_f32 += w->layers[l].ffn_norm.rows;
        } else {
            total_f32 += (uint64_t)w->layers[l].ffn_gate_inp.rows * w->layers[l].ffn_gate_inp.cols;
            total_f32 += (uint64_t)w->layers[l].ffn_gate_exps.rows * w->layers[l].ffn_gate_exps.cols;
            total_f32 += (uint64_t)w->layers[l].ffn_up_exps.rows * w->layers[l].ffn_up_exps.cols;
            total_f32 += (uint64_t)w->layers[l].ffn_down_exps.rows * w->layers[l].ffn_down_exps.cols;
            total_f32 += (uint64_t)w->layers[l].ffn_gate_shexp.rows * w->layers[l].ffn_gate_shexp.cols;
            total_f32 += (uint64_t)w->layers[l].ffn_up_shexp.rows * w->layers[l].ffn_up_shexp.cols;
            total_f32 += (uint64_t)w->layers[l].ffn_down_shexp.rows * w->layers[l].ffn_down_shexp.cols;
            total_f32 += w->layers[l].exp_probs_b.rows;
            total_f32 += w->layers[l].ffn_norm_moe.rows;
        }
    }
    fprintf(stderr, "Weights loaded: %.2f GB\n", (float)(total_f32 * 4ULL) / 1e9f);
    return true;
}

void glm_weights_free(glm_weights *w) {
    if (!w) return;
    glm_tensor_f32_free(&w->token_embd);
    glm_tensor_f32_free(&w->output);
    glm_tensor_f32_free(&w->output_norm);
    for (int l = 0; l < GLM_LAYERS; l++) {
        glm_tensor_f32_free(&w->layers[l].attn_q_a);
        glm_tensor_f32_free(&w->layers[l].attn_q_a_norm);
        glm_tensor_f32_free(&w->layers[l].attn_q_b);
        glm_tensor_f32_free(&w->layers[l].attn_k_b);
        glm_tensor_f32_free(&w->layers[l].attn_kv_a_mqa);
        glm_tensor_f32_free(&w->layers[l].attn_kv_a_norm);
        glm_tensor_f32_free(&w->layers[l].attn_v_b);
        glm_tensor_f32_free(&w->layers[l].attn_output);
        glm_tensor_f32_free(&w->layers[l].attn_norm);
        glm_tensor_f32_free(&w->layers[l].ffn_gate);
        glm_tensor_f32_free(&w->layers[l].ffn_up);
        glm_tensor_f32_free(&w->layers[l].ffn_down);
        glm_tensor_f32_free(&w->layers[l].ffn_norm);
        glm_tensor_f32_free(&w->layers[l].ffn_gate_inp);
        glm_tensor_f32_free(&w->layers[l].ffn_gate_exps);
        glm_tensor_f32_free(&w->layers[l].ffn_up_exps);
        glm_tensor_f32_free(&w->layers[l].ffn_down_exps);
        glm_tensor_f32_free(&w->layers[l].ffn_gate_shexp);
        glm_tensor_f32_free(&w->layers[l].ffn_up_shexp);
        glm_tensor_f32_free(&w->layers[l].ffn_down_shexp);
        glm_tensor_f32_free(&w->layers[l].exp_probs_b);
        glm_tensor_f32_free(&w->layers[l].ffn_norm_moe);
    }
    memset(w, 0, sizeof(*w));
}