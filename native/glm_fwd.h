#ifndef GLM_FWD_H
#define GLM_FWD_H

#include "glm_tensor.h"

#include <stdbool.h>
#include <stdint.h>

// ─── Model config (matches deepseek2 metadata) ────────────────────────────────
#define FWD_HIDDEN        2048
#define FWD_VOCAB       154880
#define FWD_HEADS          20
#define FWD_KV_HEADS        1
#define FWD_KV_Q           512   // k_lora_rank
#define FWD_KV_V           256   // v_lora_rank_mla
#define FWD_Q_PROJ        768   // q_lora_rank
#define FWD_ROPE_DIM       64
#define FWD_HEAD_DIM_Q    256   // q_lora_rank / heads = 768/20 = 38.4? No...
#define FWD_FFN_HID      10240
#define FWD_MOE_HID       1536
#define FWD_EXPERTS         64
#define FWD_MOE_K            4
#define FWD_LAYERS          47

// Derived: Q per head = q_lora_rank / num_heads_kv? No.
// DeepSeek-V2: q is also compressed via LoRA. q_a: [2048, 768], q_b: [768, 5120]
// q_lora_rank = 768, but after Q_B it's 5120 = heads * head_dim_q
// So head_dim_q = 5120 / 20 = 256. That matches!
// K/V per head: kv_lora_rank = 512, so K is 512, V is 256 per head
// But K is stored with 192 rope dim per head
#define FWD_HEAD_DIM_K     192   // key_length = 192 (RoPE part of K)
#define FWD_HEAD_DIM_V     256   // value_length_mla = 256

// RoPE freq base
#define FWD_ROPE_FREQ_BASE 1000000.0f

// ─── KV Cache entry ───────────────────────────────────────────────────────────
// For MLA with MQA, we cache: q_Compressed (seq_len, 512) and k_expanded (seq_len, 20*192)
// We store as flat arrays: [seq_len][KV_DIM] per layer

typedef struct {
    float *q;     // [max_seq, 512]   — compressed Q for re-use
    float *k_rope; // [max_seq, 20*192] — RoPE-applied K (full, all heads)
    int32_t seq_len;
    int32_t max_seq;
} kv_cache;

// ─── Runtime context (activations buffer) ─────────────────────────────────────
#define MAX_SEQ  2048
#define BATCH    1

typedef struct {
    // Hidden states: [BATCH][MAX_SEQ][HIDDEN]
    float *hidden;
    // Q/K/V activations: [BATCH][MAX_SEQ][D]
    float *q_buf;      // [MAX_SEQ][5120]
    float *k_buf;      // [MAX_SEQ][3840]
    float *v_buf;      // [MAX_SEQ][5120]
    float *attn_out;   // [MAX_SEQ][5120]
    // FFN activations
    float *ffn_buf;    // [MAX_SEQ][2048]
    float *ffn_hidden; // intermediate
    float *moe_buf;    // [MAX_SEQ][2048]
    // MoE gating
    float *gate_scores; // [MAX_SEQ][64]
    // Workspace for gemm
    float *gemm_buf;   // [MAX_SEQ][MAX(HIDDEN, VOCAB)] = [2048][154880]
    // RoPE cos/sin tables
    float *rope_cos;   // [ROPE_DIM/2]
    float *rope_sin;   // [ROPE_DIM/2]
    // KV cache per layer
    kv_cache layer_kv[FWD_LAYERS];
    int32_t max_seq;
} fwd_ctx;

bool fwd_ctx_init(fwd_ctx *ctx, int32_t max_seq);
void fwd_ctx_free(fwd_ctx *ctx);

// ─── Forward pass: full model ──────────────────────────────────────────────────
// Returns vocab logits [VOCAB] on success, NULL on failure
float *glm_forward(fwd_ctx *ctx, const glm_weights *w,
                   const int32_t *tokens, int32_t n_tokens);

// ─── Sampling ─────────────────────────────────────────────────────────────────
int32_t glm_sample_argmax(const float *logits, int32_t vocab_size);
int32_t glm_sample_temperature(const float *logits, int32_t vocab_size, float temp);

// ─── Generation loop ──────────────────────────────────────────────────────────
// Fills `output_tokens` with generated tokens (up to max_new_tokens).
// Returns actual number of tokens generated (excluding input).
int32_t glm_generate(fwd_ctx *ctx, const glm_weights *w,
                     const int32_t *input_tokens, int32_t n_input,
                     int32_t *output_tokens, int32_t max_new_tokens,
                     float temperature);

#endif // GLM_FWD_H