#include "glm_gguf.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    glm_gguf_model *m;
    uint64_t pos;
} reader;

static void set_error(glm_gguf_model *m, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(m->error, sizeof(m->error), fmt, ap);
    va_end(ap);
}

static bool read_exact(reader *r, void *dst, size_t len) {
    if (fread(dst, 1, len, r->m->fp) != len) {
        set_error(r->m, "short read at byte %" PRIu64, r->pos);
        return false;
    }
    r->pos += (uint64_t)len;
    return true;
}

static bool read_u32(reader *r, uint32_t *out) {
    return read_exact(r, out, sizeof(*out));
}

static bool read_u64(reader *r, uint64_t *out) {
    return read_exact(r, out, sizeof(*out));
}

static bool read_f32(reader *r, float *out) {
    return read_exact(r, out, sizeof(*out));
}

static bool read_f64(reader *r, double *out) {
    return read_exact(r, out, sizeof(*out));
}

static bool skip_bytes(reader *r, uint64_t len) {
    if (len > (uint64_t)LONG_MAX) {
        set_error(r->m, "skip too large at byte %" PRIu64, r->pos);
        return false;
    }
    if (fseek(r->m->fp, (long)len, SEEK_CUR) != 0) {
        set_error(r->m, "seek failed at byte %" PRIu64 ": %s", r->pos, strerror(errno));
        return false;
    }
    r->pos += len;
    return true;
}

static bool read_string(reader *r, char **out) {
    uint64_t len = 0;
    if (!read_u64(r, &len)) return false;
    if (len > (UINT64_C(1) << 32)) {
        set_error(r->m, "string too large at byte %" PRIu64, r->pos);
        return false;
    }

    char *s = calloc((size_t)len + 1, 1);
    if (!s) {
        set_error(r->m, "out of memory reading string");
        return false;
    }
    if (len && !read_exact(r, s, (size_t)len)) {
        free(s);
        return false;
    }
    *out = s;
    return true;
}

static bool skip_scalar(reader *r, uint32_t type);

static bool skip_array(reader *r) {
    uint32_t elem_type = 0;
    uint64_t len = 0;
    if (!read_u32(r, &elem_type)) return false;
    if (!read_u64(r, &len)) return false;

    for (uint64_t i = 0; i < len; i++) {
        if (!skip_scalar(r, elem_type)) return false;
    }
    return true;
}

static bool skip_scalar(reader *r, uint32_t type) {
    switch (type) {
        case GLM_GGUF_VALUE_UINT8:
        case GLM_GGUF_VALUE_INT8:
        case GLM_GGUF_VALUE_BOOL:
            return skip_bytes(r, 1);
        case GLM_GGUF_VALUE_UINT16:
        case GLM_GGUF_VALUE_INT16:
            return skip_bytes(r, 2);
        case GLM_GGUF_VALUE_UINT32:
        case GLM_GGUF_VALUE_INT32:
        case GLM_GGUF_VALUE_FLOAT32:
            return skip_bytes(r, 4);
        case GLM_GGUF_VALUE_UINT64:
        case GLM_GGUF_VALUE_INT64:
        case GLM_GGUF_VALUE_FLOAT64:
            return skip_bytes(r, 8);
        case GLM_GGUF_VALUE_STRING: {
            char *tmp = NULL;
            bool ok = read_string(r, &tmp);
            free(tmp);
            return ok;
        }
        case GLM_GGUF_VALUE_ARRAY:
            return skip_array(r);
        default:
            set_error(r->m, "unknown GGUF value type %u at byte %" PRIu64, type, r->pos);
            return false;
    }
}

static uint64_t align_u64(uint64_t x, uint32_t alignment) {
    if (alignment == 0) alignment = 32;
    const uint64_t a = alignment;
    return ((x + a - 1) / a) * a;
}

bool glm_gguf_open(glm_gguf_model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->alignment = 32;
    m->fp = fopen(path, "rb");
    if (!m->fp) {
        set_error(m, "failed to open %s: %s", path, strerror(errno));
        return false;
    }

    reader r = { .m = m, .pos = 0 };
    uint32_t magic = 0;
    if (!read_u32(&r, &magic)) return false;
    if (magic != GLM_GGUF_MAGIC) {
        set_error(m, "not a GGUF file");
        return false;
    }
    if (!read_u32(&r, &m->version)) return false;
    if (!read_u64(&r, &m->n_tensors)) return false;
    if (!read_u64(&r, &m->n_kv)) return false;

    if (m->n_kv > UINT64_C(1000000) || m->n_tensors > UINT64_C(1000000)) {
        set_error(m, "unreasonable GGUF table sizes");
        return false;
    }

    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if ((m->n_kv && !m->kv) || (m->n_tensors && !m->tensors)) {
        set_error(m, "out of memory allocating GGUF tables");
        return false;
    }

    for (uint64_t i = 0; i < m->n_kv; i++) {
        glm_gguf_kv *kv = &m->kv[i];
        if (!read_string(&r, &kv->key)) return false;
        if (!read_u32(&r, &kv->type)) return false;
        kv->value_pos = r.pos;
        if (strcmp(kv->key, "general.alignment") == 0 && kv->type == GLM_GGUF_VALUE_UINT32) {
            long cur = ftell(m->fp);
            if (cur >= 0) {
                uint32_t alignment = 0;
                if (read_u32(&r, &alignment) && alignment) m->alignment = alignment;
                if (fseek(m->fp, cur, SEEK_SET) == 0) r.pos = (uint64_t)cur;
            }
        }
        if (!skip_scalar(&r, kv->type)) return false;
    }

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        glm_gguf_tensor *t = &m->tensors[i];
        if (!read_string(&r, &t->name)) return false;
        if (!read_u32(&r, &t->ndim)) return false;
        if (t->ndim == 0 || t->ndim > GLM_GGUF_MAX_DIMS) {
            set_error(m, "tensor %s has invalid rank %u", t->name, t->ndim);
            return false;
        }
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!read_u64(&r, &t->dim[d])) return false;
        }
        if (!read_u32(&r, &t->type)) return false;
        if (!read_u64(&r, &t->offset)) return false;
    }

    m->tensor_data_offset = align_u64(r.pos, m->alignment);
    return true;
}

void glm_gguf_close(glm_gguf_model *m) {
    if (!m) return;
    if (m->fp) fclose(m->fp);
    for (uint64_t i = 0; i < m->n_kv; i++) free(m->kv[i].key);
    for (uint64_t i = 0; i < m->n_tensors; i++) free(m->tensors[i].name);
    free(m->kv);
    free(m->tensors);
    memset(m, 0, sizeof(*m));
}

const glm_gguf_kv *glm_gguf_find_kv(const glm_gguf_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (strcmp(m->kv[i].key, key) == 0) return &m->kv[i];
    }
    return NULL;
}

static bool seek_value(const glm_gguf_model *m, const glm_gguf_kv *kv) {
    return kv && fseek(m->fp, (long)kv->value_pos, SEEK_SET) == 0;
}

bool glm_gguf_get_string(const glm_gguf_model *m, const char *key, char **out) {
    const glm_gguf_kv *kv = glm_gguf_find_kv(m, key);
    if (!kv || kv->type != GLM_GGUF_VALUE_STRING || !seek_value(m, kv)) return false;
    reader r = { .m = (glm_gguf_model *)m, .pos = kv->value_pos };
    return read_string(&r, out);
}

bool glm_gguf_get_u32(const glm_gguf_model *m, const char *key, uint32_t *out) {
    const glm_gguf_kv *kv = glm_gguf_find_kv(m, key);
    if (!kv || !seek_value(m, kv)) return false;
    reader r = { .m = (glm_gguf_model *)m, .pos = kv->value_pos };
    if (kv->type == GLM_GGUF_VALUE_UINT32) return read_u32(&r, out);
    if (kv->type == GLM_GGUF_VALUE_UINT64) {
        uint64_t v = 0;
        if (!read_u64(&r, &v) || v > UINT32_MAX) return false;
        *out = (uint32_t)v;
        return true;
    }
    return false;
}

bool glm_gguf_get_u64(const glm_gguf_model *m, const char *key, uint64_t *out) {
    const glm_gguf_kv *kv = glm_gguf_find_kv(m, key);
    if (!kv || !seek_value(m, kv)) return false;
    reader r = { .m = (glm_gguf_model *)m, .pos = kv->value_pos };
    if (kv->type == GLM_GGUF_VALUE_UINT64) return read_u64(&r, out);
    if (kv->type == GLM_GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!read_u32(&r, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

bool glm_gguf_get_f32(const glm_gguf_model *m, const char *key, float *out) {
    const glm_gguf_kv *kv = glm_gguf_find_kv(m, key);
    if (!kv || !seek_value(m, kv)) return false;
    reader r = { .m = (glm_gguf_model *)m, .pos = kv->value_pos };
    if (kv->type == GLM_GGUF_VALUE_FLOAT32) return read_f32(&r, out);
    if (kv->type == GLM_GGUF_VALUE_FLOAT64) {
        double v = 0;
        if (!read_f64(&r, &v)) return false;
        *out = (float)v;
        return true;
    }
    return false;
}

bool glm_gguf_get_bool(const glm_gguf_model *m, const char *key, bool *out) {
    const glm_gguf_kv *kv = glm_gguf_find_kv(m, key);
    if (!kv || kv->type != GLM_GGUF_VALUE_BOOL || !seek_value(m, kv)) return false;
    uint8_t v = 0;
    reader r = { .m = (glm_gguf_model *)m, .pos = kv->value_pos };
    if (!read_exact(&r, &v, 1)) return false;
    *out = v != 0;
    return true;
}

const char *glm_gguf_value_type_name(uint32_t type) {
    switch (type) {
        case GLM_GGUF_VALUE_UINT8: return "uint8";
        case GLM_GGUF_VALUE_INT8: return "int8";
        case GLM_GGUF_VALUE_UINT16: return "uint16";
        case GLM_GGUF_VALUE_INT16: return "int16";
        case GLM_GGUF_VALUE_UINT32: return "uint32";
        case GLM_GGUF_VALUE_INT32: return "int32";
        case GLM_GGUF_VALUE_FLOAT32: return "float32";
        case GLM_GGUF_VALUE_BOOL: return "bool";
        case GLM_GGUF_VALUE_STRING: return "string";
        case GLM_GGUF_VALUE_ARRAY: return "array";
        case GLM_GGUF_VALUE_UINT64: return "uint64";
        case GLM_GGUF_VALUE_INT64: return "int64";
        case GLM_GGUF_VALUE_FLOAT64: return "float64";
        default: return "unknown";
    }
}

const char *glm_gguf_tensor_type_name(uint32_t type) {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 31: return "TQ1_0";
        case 32: return "TQ2_0";
        default: return "unknown";
    }
}

uint64_t glm_gguf_tensor_elems(const glm_gguf_tensor *t) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->ndim; i++) {
        if (t->dim[i] == 0 || n > UINT64_MAX / t->dim[i]) return 0;
        n *= t->dim[i];
    }
    return n;
}

void glm_gguf_print_summary(const glm_gguf_model *m, FILE *fp) {
    fprintf(fp, "GGUF version:       %u\n", m->version);
    fprintf(fp, "metadata entries:   %" PRIu64 "\n", m->n_kv);
    fprintf(fp, "tensor count:       %" PRIu64 "\n", m->n_tensors);
    fprintf(fp, "alignment:          %u\n", m->alignment);
    fprintf(fp, "tensor data offset: %" PRIu64 "\n", m->tensor_data_offset);
}
