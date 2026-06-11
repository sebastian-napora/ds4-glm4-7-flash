#include "glm_fwd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ─── Memory management ───────────────────────────────────────────────────────
static float *alloc_f(size_t n) {
    float *p = (float *)calloc(n, sizeof(float));
    if (!p) { fprintf(stderr, "glm_fwd OOM: %.1f GB\n", (float)n * 4.0f / 1e9f); }
    return p;
}

// ─── Math primitives ─────────────────────────────────────────────────────────

// RMSNorm: y[i] = x[i] * (w[i] / sqrt(sum(x^2)/n + eps))
static void rmsnorm(float *y, const float *x, const float *w,
                    uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
}

// GEMV: y[N] = x[K] @ W[K,N]  (W stored row-major as [K,N])
static void gemv(float *y, const float *x, const float *W,
                 uint32_t K, uint32_t N) {
    for (uint32_t n = 0; n < N; n++) {
        float s = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            s += x[k] * W[(uint64_t)k * N + n];
        }
        y[n] = s;
    }
}

// GEMM: C[M,N] = A[M,K] @ B[K,N]  (all row-major, B transposed)
static void gemm(float *C, const float *A, const float *B,
                 uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; m++) {
        const float *A_row = A + (uint64_t)m * K;
        float *C_row = C + (uint64_t)m * N;
        for (uint32_t n = 0; n < N; n++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                s += A_row[k] * B[(uint64_t)k * N + n];
            }
            C_row[n] = s;
        }
    }
}

// y += x  (in-place element-wise add)
static void add_inplace(float *y, const float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) y[i] += x[i];
}

// SiLU: y[i] = x[i] / (1 + exp(-x[i]))
static void silu(float *y, const float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}

// softmax over array of logits (in-place)
static void softmax(float *y, uint32_t n) {
    float mx = y[0];
    for (uint32_t i = 1; i < n; i++) if (y[i] > mx) mx = y[i];
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += expf(y[i] - mx);
    ss = 1.0f / ss;
    for (uint32_t i = 0; i < n; i++) y[i] = expf(y[i] - mx) * ss;
}

// ─── Context management ──────────────────────────────────────────────────────
bool fwd_ctx_init(fwd_ctx *ctx, int32_t max_seq) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->max_seq = max_seq;

    ctx->hidden      = alloc_f((size_t)max_seq * FWD_HIDDEN);
    ctx->q_buf       = alloc_f((size_t)max_seq * 5120);    // [max_seq][20][256]
    ctx->k_buf       = alloc_f((size_t)max_seq * 3840);    // [max_seq][20][192]
    ctx->v_buf       = alloc_f((size_t)max_seq * 5120);    // [max_seq][20][256]
    ctx->attn_out    = alloc_f((size_t)max_seq * 5120);
    ctx->ffn_buf     = alloc_f((size_t)max_seq * FWD_HIDDEN);
    ctx->ffn_hidden  = alloc_f((size_t)max_seq * 10240);
    ctx->moe_buf     = alloc_f((size_t)max_seq * FWD_HIDDEN);
    ctx->gate_scores = alloc_f((size_t)max_seq * FWD_EXPERTS);
    ctx->gemm_buf    = alloc_f((size_t)max_seq * FWD_VOCAB);
    ctx->rope_cos    = alloc_f(FWD_ROPE_DIM / 2);
    ctx->rope_sin    = alloc_f(FWD_ROPE_DIM / 2);

    for (int l = 0; l < FWD_LAYERS; l++) {
        ctx->layer_kv[l].q      = alloc_f((size_t)max_seq * 512);  // compressed Q
        ctx->layer_kv[l].k_rope = alloc_f((size_t)max_seq * 3840); // RoPE K per head
        ctx->layer_kv[l].max_seq = max_seq;
        ctx->layer_kv[l].seq_len = 0;
    }

    if (!ctx->hidden || !ctx->q_buf || !ctx->k_buf || !ctx->v_buf ||
        !ctx->attn_out || !ctx->ffn_buf || !ctx->ffn_hidden ||
        !ctx->moe_buf || !ctx->gate_scores || !ctx->gemm_buf) {
        fwd_ctx_free(ctx);
        return false;
    }
    return true;
}

void fwd_ctx_free(fwd_ctx *ctx) {
    if (!ctx) return;
    free(ctx->hidden);
    free(ctx->q_buf);
    free(ctx->k_buf);
    free(ctx->v_buf);
    free(ctx->attn_out);
    free(ctx->ffn_buf);
    free(ctx->ffn_hidden);
    free(ctx->moe_buf);
    free(ctx->gate_scores);
    free(ctx->gemm_buf);
    free(ctx->rope_cos);
    free(ctx->rope_sin);
    for (int l = 0; l < FWD_LAYERS; l++) {
        free(ctx->layer_kv[l].q);
        free(ctx->layer_kv[l].k_rope);
    }
    memset(ctx, 0, sizeof(*ctx));
}

// ─── Helper: apply RoPE to one k/c head vector ───────────────────────────────
// k: [192] for K head or [256] for Q head
// RoPE dims 0..31 (half=FWD_ROPE_DIM/2=32) are rotated.
// Position pos (0-based), freq_base = 1e6.
static void rope_1d(float *v, int32_t pos, uint32_t head_dim, float freq_base) {
    uint32_t half = FWD_ROPE_DIM / 2;
    if (half > head_dim) half = head_dim;
    for (uint32_t i = 0; i < half; i++) {
        float theta = 1.0f / powf(freq_base, (float)(2 * i) / (float)FWD_ROPE_DIM);
        float ct = cosf((float)pos * theta);
        float st = sinf((float)pos * theta);
        float x0 = v[i];
        float x1 = v[i + half];
        v[i]       = x0 * ct - x1 * st;
        v[i + half] = x0 * st + x1 * ct;
    }
}

// ─── MoE FFN (layers 1-46) ──────────────────────────────────────────────────
// ffn_gate_inp:   [2048, 64]     — top-k router
// ffn_gate_exps:  [131072, 1536]  — interleaved: row e*2048+f = expert e, feature f
// ffn_up_exps:    [131072, 1536]  — same layout
// ffn_down_exps:  [98304, 2048]   — row e*1536+m = expert e, intermediate m
// ffn_gate_shexp/up_shexp/down_shexp: shared expert [2048, 1536] / [1536, 2048]
static void moe_ffn(float *out, const float *x,
                    const struct layer_weights *lw,
                    float *tmp, uint32_t n_active) {
    (void)tmp;
    uint32_t H = FWD_HIDDEN;
    uint32_t M = FWD_MOE_HID;
    uint32_t E = FWD_EXPERTS;
    uint32_t F = H;  // F = hidden dim

    // Gate scores: [64] = x @ ffn_gate_inp
    gemv(tmp, x, lw->ffn_gate_inp.data, H, E);

    // Top-k
    float top_k_scores[4];
    int   top_k_idx[4];
    for (uint32_t kk = 0; kk < n_active; kk++) { top_k_scores[kk] = -1e9f; top_k_idx[kk] = -1; }
    for (uint32_t e = 0; e < E; e++) {
        float s = tmp[e];
        if (s > top_k_scores[n_active - 1]) {
            top_k_scores[n_active - 1] = s;
            top_k_idx[n_active - 1] = (int)e;
            for (int kk = (int)n_active - 2; kk >= 0; kk--) {
                if (top_k_scores[kk + 1] > top_k_scores[kk]) {
                    float ts = top_k_scores[kk]; top_k_scores[kk] = top_k_scores[kk + 1]; top_k_scores[kk + 1] = ts;
                    int   ti = top_k_idx[kk];   top_k_idx[kk]   = top_k_idx[kk + 1];   top_k_idx[kk + 1]   = ti;
                } else break;
            }
        }
    }

    // Softmax over top-k
    float mx = top_k_scores[0];
    for (uint32_t kk = 1; kk < n_active; kk++) if (top_k_scores[kk] > mx) mx = top_k_scores[kk];
    float ss = 0.0f;
    for (uint32_t kk = 0; kk < n_active; kk++) ss += expf(top_k_scores[kk] - mx);
    float ssf = 1.0f / ss;

    memset(out, 0, H * sizeof(float));
    float up_buf[1536];
    float gate_buf[1536];

    for (uint32_t kk = 0; kk < n_active; kk++) {
        int e = top_k_idx[kk];
        float weight = expf(top_k_scores[kk] - mx) * ssf;

        // gate: [1536] = x @ ffn_gate_exps row (e*F)
        gemv(up_buf, x, lw->ffn_gate_exps.data + (uint64_t)e * F, F, M);
        // up:   [1536] = x @ ffn_up_exps row (e*F)
        gemv(gate_buf, x, lw->ffn_up_exps.data + (uint64_t)e * F, F, M);
        // silu(gate) * up -> gate_buf (in-place)
        for (uint32_t i = 0; i < M; i++) {
            float g = gate_buf[i];
            gate_buf[i] = (g / (1.0f + expf(-g))) * up_buf[i];
        }
        // down: out += weight * gate_buf @ ffn_down_exps row (e*M)
        const float *W_down = lw->ffn_down_exps.data + (uint64_t)e * M;
        for (uint32_t h = 0; h < H; h++) {
            float s = 0.0f;
            for (uint32_t mi = 0; mi < M; mi++) {
                s += gate_buf[mi] * W_down[mi * H + h];
            }
            out[h] += weight * s;
        }
    }

    // Shared expert (always active)
    gemv(up_buf, x, lw->ffn_up_shexp.data, H, M);
    gemv(gate_buf, x, lw->ffn_gate_shexp.data, H, M);
    for (uint32_t i = 0; i < M; i++) {
        float g = gate_buf[i];
        gate_buf[i] = (g / (1.0f + expf(-g))) * up_buf[i];
    }
    // down_shared: [2048] += gate_buf @ ffn_down_shexp [1536, 2048]
    gemv(out + H, gate_buf, lw->ffn_down_shexp.data, M, H);
    for (uint32_t h = 0; h < H; h++) out[h] += out[H + h];
}

// ─── Dense FFN (layer 0) ─────────────────────────────────────────────────────
// ffn_gate: [2048, 10240], ffn_up: [2048, 10240], ffn_down: [10240, 2048]
static void dense_ffn(float *out, const float *x,
                      const struct layer_weights *lw,
                      float *tmp) {
    // tmp[0..10239]:  x @ ffn_up   -> [10240]
    gemv(tmp, x, lw->ffn_up.data, FWD_HIDDEN, FWD_FFN_HID);
    // tmp[10240..20479]: x @ ffn_gate -> [10240]
    gemv(tmp + FWD_FFN_HID, x, lw->ffn_gate.data, FWD_HIDDEN, FWD_FFN_HID);
    // silu(gate) * up -> tmp[0..10239] in-place
    for (uint32_t i = 0; i < FWD_FFN_HID; i++) {
        float g = tmp[FWD_FFN_HID + i];
        tmp[i] = (g / (1.0f + expf(-g))) * tmp[i];
    }
    // out: [2048] = tmp @ ffn_down
    gemv(out, tmp, lw->ffn_down.data, FWD_FFN_HID, FWD_HIDDEN);
}

// ─── MLA Attention (prefill, all tokens at once) ─────────────────────────────
// Layout per token:
//   q_buf:  [20][256]  = ctx->q_buf[s*5120 + h*256]
//   k_buf:  [20][192]  = ctx->k_buf[s*3840 + h*192]
//   v_buf:  [20][256]  = ctx->v_buf[s*5120 + h*256]
static void mla_attention_layer(fwd_ctx *ctx,
                                 const struct layer_weights *lw,
                                 int32_t n_tokens) {
    // Per-token Q/K/V computation
    for (int32_t s = 0; s < n_tokens; s++) {
        float *x = ctx->hidden + (uint64_t)s * FWD_HIDDEN;
        float *q_s = ctx->q_buf + (uint64_t)s * 5120;
        float *k_s = ctx->k_buf + (uint64_t)s * 3840;
        float *v_s = ctx->v_buf + (uint64_t)s * 5120;

        // Q path: Q_a = x @ W_q_a -> [768], norm, Q_b -> [5120]
        float q_a[768];
        gemv(q_a, x, lw->attn_q_a.data, FWD_HIDDEN, FWD_Q_PROJ);
        float q_a_n[768];
        rmsnorm(q_a_n, q_a, lw->attn_q_a_norm.data, FWD_Q_PROJ, 1e-5f);
        gemv(q_s, q_a_n, lw->attn_q_b.data, FWD_Q_PROJ, 5120);

        // KV path: KV_a = x @ W_kv_a -> [576]
        float kv_a[576];
        gemv(kv_a, x, lw->attn_kv_a_mqa.data, FWD_HIDDEN, 576);
        float kv_a_n[512];
        rmsnorm(kv_a_n, kv_a, lw->attn_kv_a_norm.data, 512, 1e-5f);

        // K_c = kv_a_n[192:] @ W_k_b [320, 20, 192] -> [20, 192]
        // K has 192 dims per head; kv_a_n[192:] has 320 dims
        // attn_k_b layout: [320][192][20] -> our [20*192=3840][320]
        for (int h = 0; h < 20; h++) {
            const float *W_kb = lw->attn_k_b.data + (uint64_t)h * 320; // [320]
            float *k_ch = k_s + (uint64_t)h * 192;
            for (int kk = 0; kk < 192; kk++) {
                float s = 0.0f;
                for (int ki = 0; ki < 320; ki++) {
                    s += kv_a_n[192 + ki] * W_kb[ki * 192 + kk];
                }
                k_ch[kk] = s;
            }
            // Apply RoPE to first 64 dims of each head
            rope_1d(k_ch, (int)s, 192, FWD_ROPE_FREQ_BASE);
        }

        // V_c = kv_a_n[256:] @ W_v_b [256, 20, 256] -> [20, 256]
        // attn_v_b layout: [256][256][20] -> our [20*256=5120][256]
        for (int h = 0; h < 20; h++) {
            const float *W_vb = lw->attn_v_b.data + (uint64_t)h * 256; // [256]
            float *v_ch = v_s + (uint64_t)h * 256;
            for (int kk = 0; kk < 256; kk++) {
                float s = 0.0f;
                for (int ki = 0; ki < 256; ki++) {
                    s += kv_a_n[256 + ki] * W_vb[ki * 256 + kk];
                }
                v_ch[kk] = s;
            }
        }
    }

    // ── Full attention O(n²) for prefill ─────────────────────────────────
    // attn_out: [n_tokens][5120]
    memset(ctx->attn_out, 0, (size_t)n_tokens * 5120 * sizeof(float));

    float scale = 1.0f / sqrtf(256.0f);  // 1/sqrt(head_dim_q)

    for (int h = 0; h < 20; h++) {
        // Q head h: [n_tokens][256]
        const float *q_h = ctx->q_buf + (uint64_t)h * n_tokens * 256;
        // K head h: [n_tokens][192]
        const float *k_h = ctx->k_buf + (uint64_t)h * n_tokens * 192;
        // V head h: [n_tokens][256]
        const float *v_h = ctx->v_buf + (uint64_t)h * n_tokens * 256;
        // Out head h: [n_tokens][256]  (stored in attn_out)
        float *out_h = ctx->attn_out + (uint64_t)h * n_tokens * 256;

        for (int qi = 0; qi < n_tokens; qi++) {
            const float *q_vec = q_h + (uint64_t)qi * 256;

            // Scores for this query token against all key tokens
            float scores[512];
            for (int ki = 0; ki < n_tokens; ki++) {
                const float *k_vec = k_h + (uint64_t)ki * 192;
                float dot = 0.0f;
                // Only dot over the portion we actually have (192 dims for K)
                for (int d = 0; d < 192; d++) dot += q_vec[d] * k_vec[d];
                scores[ki] = dot * scale;
            }

            // Softmax
            float mx = scores[0];
            for (int ki = 1; ki < n_tokens; ki++) if (scores[ki] > mx) mx = scores[ki];
            float ss = 0.0f;
            for (int ki = 0; ki < n_tokens; ki++) ss += expf(scores[ki] - mx);
            ss = 1.0f / ss;
            for (int ki = 0; ki < n_tokens; ki++) scores[ki] = expf(scores[ki] - mx) * ss;

            // Accumulate into out_h[qi]
            float *out_vec = out_h + (uint64_t)qi * 256;
            for (int ki = 0; ki < n_tokens; ki++) {
                const float *v_vec = v_h + (uint64_t)ki * 256;
                float w = scores[ki];
                for (int d = 0; d < 256; d++) out_vec[d] += w * v_vec[d];
            }
        }
    }

    // Project attention output back to hidden dim and add residual
    // attn_out: [n_tokens][5120], W_out: [5120, 2048]
    for (int s = 0; s < n_tokens; s++) {
        float *src = ctx->attn_out + (uint64_t)s * 5120;
        float *dst = ctx->q_buf + (uint64_t)s * 5120;  // reuse q_buf as temp output
        gemv(dst, src, lw->attn_output.data, 5120, FWD_HIDDEN);
        // Add residual: hidden += attn_output
        float *hidden_row = ctx->hidden + (uint64_t)s * FWD_HIDDEN;
        add_inplace(hidden_row, dst, FWD_HIDDEN);
    }
}

// ─── One full forward pass ───────────────────────────────────────────────────
float *glm_forward(fwd_ctx *ctx, const glm_weights *w,
                    const int32_t *tokens, int32_t n_tokens) {
    if (!ctx || !w || !tokens || n_tokens <= 0 || n_tokens > ctx->max_seq) {
        return NULL;
    }

    // Step 1: Token embedding
    for (int32_t t = 0; t < n_tokens; t++) {
        int32_t tok = tokens[t];
        if ((uint32_t)tok >= FWD_VOCAB) tok = 0;
        const float *emb = w->token_embd.data + (uint64_t)tok * FWD_HIDDEN;
        float *h = ctx->hidden + (uint64_t)t * FWD_HIDDEN;
        memcpy(h, emb, FWD_HIDDEN * sizeof(float));
    }

    // Step 2: Transformer layers
    for (int layer = 0; layer < FWD_LAYERS; layer++) {
        const struct layer_weights *lw = &w->layers[layer];
        int is_moe = (layer > 0);

        // Pre-norm before attention
        for (int s = 0; s < n_tokens; s++) {
            float *h_row = ctx->hidden + (uint64_t)s * FWD_HIDDEN;
            float *h_norm = ctx->gemm_buf + (uint64_t)s * FWD_HIDDEN;
            rmsnorm(h_norm, h_row, lw->attn_norm.data, FWD_HIDDEN, 1e-5f);
            // Write normed into hidden for attention
            memcpy(h_row, h_norm, FWD_HIDDEN * sizeof(float));
        }

        // MLA attention
        mla_attention_layer(ctx, lw, n_tokens);

        // Pre-norm before FFN
        for (int s = 0; s < n_tokens; s++) {
            float *h_row = ctx->hidden + (uint64_t)s * FWD_HIDDEN;
            float *h_norm = ctx->gemm_buf + (uint64_t)s * FWD_HIDDEN;
            const float *norm_w = is_moe ? lw->ffn_norm_moe.data : lw->ffn_norm.data;
            rmsnorm(h_norm, h_row, norm_w, FWD_HIDDEN, 1e-5f);
            memcpy(h_row, h_norm, FWD_HIDDEN * sizeof(float));
        }

        // FFN
        for (int s = 0; s < n_tokens; s++) {
            float *x = ctx->hidden + (uint64_t)s * FWD_HIDDEN;
            float *ffn_out = ctx->ffn_buf + (uint64_t)s * FWD_HIDDEN;
            if (is_moe) {
                moe_ffn(ffn_out, x, lw, ctx->ffn_hidden, FWD_MOE_K);
            } else {
                dense_ffn(ffn_out, x, lw, ctx->ffn_hidden);
            }
            add_inplace(x, ffn_out, FWD_HIDDEN);
        }
    }

    // Step 3: Final RMSNorm + LM head
    float *last_h = ctx->hidden + (uint64_t)(n_tokens - 1) * FWD_HIDDEN;
    float *normed = ctx->gemm_buf;  // reuse start of gemm_buf
    rmsnorm(normed, last_h, w->output_norm.data, FWD_HIDDEN, 1e-5f);
    gemv(ctx->gemm_buf + FWD_HIDDEN, normed, w->output.data, FWD_HIDDEN, FWD_VOCAB);

    return ctx->gemm_buf + FWD_HIDDEN;
}

// ─── Sampling ────────────────────────────────────────────────────────────────
int32_t glm_sample_argmax(const float *logits, int32_t vocab_size) {
    int32_t best = 0;
    float mx = logits[0];
    for (int32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > mx) { mx = logits[i]; best = i; }
    }
    return best;
}

int32_t glm_sample_temperature(const float *logits, int32_t vocab_size, float temp) {
    if (temp <= 0.0f) return glm_sample_argmax(logits, vocab_size);
    // Copy logits / temp and softmax in place
    float *probs = (float *)malloc((size_t)vocab_size * sizeof(float));
    if (!probs) return glm_sample_argmax(logits, vocab_size);
    for (int32_t i = 0; i < vocab_size; i++) probs[i] = logits[i] / temp;
    softmax(probs, (uint32_t)vocab_size);
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int32_t i = 0; i < vocab_size; i++) {
        cum += probs[i];
        if (r <= cum) { free(probs); return i; }
    }
    free(probs);
    return vocab_size - 1;
}

// ─── Generation loop ──────────────────────────────────────────────────────────
int32_t glm_generate(fwd_ctx *ctx, const glm_weights *w,
                     const int32_t *input_tokens, int32_t n_input,
                     int32_t *output_tokens, int32_t max_new_tokens,
                     float temperature) {
    if (!ctx || !w || !input_tokens || !output_tokens || max_new_tokens <= 0) return 0;

    int32_t total_tokens[2048];
    memcpy(total_tokens, input_tokens, (size_t)n_input * sizeof(int32_t));
    int32_t n_total = n_input;
    int32_t generated = 0;

    while (generated < max_new_tokens && n_total < ctx->max_seq) {
        float *logits = glm_forward(ctx, w, total_tokens, n_total);
        if (!logits) return generated;

        int32_t next;
        if (temperature > 0.0f) {
            next = glm_sample_temperature(logits, FWD_VOCAB, temperature);
        } else {
            next = glm_sample_argmax(logits, FWD_VOCAB);
        }

        if (next == 0 || next >= (int32_t)FWD_VOCAB) break;
        output_tokens[generated++] = next;
        total_tokens[n_total++] = next;
    }

    return generated;
}