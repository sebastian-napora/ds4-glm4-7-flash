#ifndef GLM_TENSOR_H
#define GLM_TENSOR_H

#include "glm_gguf.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define T_F32    0
#define T_F16    1
#define T_Q8_0   8
#define T_Q6_K   14

#define GLM_HIDDEN     2048
#define GLM_VOCAB     154880
#define GLM_HEADS       20
#define GLM_KV_HEADS     1
#define GLM_ROPE_DIM    64
#define GLM_Q_LORA     768
#define GLM_KV_LORA    512
#define GLM_FFN_HID   10240
#define GLM_MOE_FFN   1536
#define GLM_EXPERTS      64
#define GLM_MOE_K         4
#define GLM_LAYERS       47

// ─── F32 tensor (dequantized; quantized data dequantized on load) ───────────────
typedef struct {
    float *data;          // [rows][cols] or [elems]
    uint32_t rows;
    uint32_t cols;
    bool is_quant;
} glm_tensor_f32;

void glm_tensor_f32_free(glm_tensor_f32 *t);

// ─── Full model weights ────────────────────────────────────────────────────────
typedef struct {
    // Global
    glm_tensor_f32 token_embd;     // [2048, 154880]
    glm_tensor_f32 output;        // [2048, 154880]
    glm_tensor_f32 output_norm;   // [2048]

    struct layer_weights {
        // Attention (shared by dense + MoE layers)
        glm_tensor_f32 attn_q_a;          // [2048, 768]
        glm_tensor_f32 attn_q_a_norm;     // [768]
        glm_tensor_f32 attn_q_b;          // [768, 5120]
        glm_tensor_f32 attn_k_b;          // [20, 192, 512]  (3D flattened)
        glm_tensor_f32 attn_kv_a_mqa;     // [2048, 576]
        glm_tensor_f32 attn_kv_a_norm;    // [512]
        glm_tensor_f32 attn_v_b;          // [20, 512, 256]  (3D flattened)
        glm_tensor_f32 attn_output;       // [5120, 2048]
        glm_tensor_f32 attn_norm;         // [2048]

        // FFN — dense (layer 0)
        glm_tensor_f32 ffn_gate;          // [2048, 10240]
        glm_tensor_f32 ffn_up;            // [2048, 10240]
        glm_tensor_f32 ffn_down;          // [10240, 2048]
        glm_tensor_f32 ffn_norm;          // [2048]

        // MoE — layers 1-46
        glm_tensor_f32 ffn_gate_inp;     // [2048, 64]
        glm_tensor_f32 ffn_gate_exps;    // [2048, 1536, 64]  (3D)
        glm_tensor_f32 ffn_up_exps;      // [2048, 1536, 64]  (3D)
        glm_tensor_f32 ffn_down_exps;    // [1536, 2048, 64]  (3D)
        glm_tensor_f32 ffn_gate_shexp;   // [2048, 1536]
        glm_tensor_f32 ffn_up_shexp;     // [2048, 1536]
        glm_tensor_f32 ffn_down_shexp;   // [1536, 2048]
        glm_tensor_f32 exp_probs_b;      // [64]
        glm_tensor_f32 ffn_norm_moe;     // [2048]
    } layers[GLM_LAYERS];
} glm_weights;

bool glm_weights_load(glm_weights *w, const glm_gguf_model *m);
void glm_weights_free(glm_weights *w);

#endif // GLM_TENSOR_H