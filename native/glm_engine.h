#ifndef GLM_ENGINE_H
#define GLM_ENGINE_H

#include "glm_gguf.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    char *architecture;
    char *name;
    char *chat_template;

    uint32_t block_count;
    uint64_t context_length;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;
    uint32_t expert_count;
    uint32_t expert_used_count;

    uint64_t expert_tensor_count;
    uint64_t attention_tensor_count;
    uint64_t block_tensor_count;
} glm_model_info;

typedef struct {
    char error[256];
    char model_path[1024];
    glm_gguf_model gguf;
    glm_model_info info;
} glm_engine;

bool glm_engine_open(glm_engine *e, const char *model_path);
void glm_engine_close(glm_engine *e);
void glm_engine_print_summary(const glm_engine *e, FILE *fp);
const char *glm_engine_model_id(const glm_engine *e);

bool glm_engine_generation_available(const glm_engine *e);
int glm_engine_generate_text(glm_engine *e, const char *prompt, char *out, size_t out_cap);

#endif
