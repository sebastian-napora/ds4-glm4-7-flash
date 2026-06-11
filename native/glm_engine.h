#ifndef GLM_ENGINE_H
#define GLM_ENGINE_H

#include "glm_gguf.h"
#include "glm_tensor.h"
#include "glm_fwd.h"

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
    glm_weights weights;
    bool weights_loaded;

    /* ── Tokenizer subprocess (Python HF AutoTokenizer) ── */
    void *tok_stdin;   /* FILE* — write end of tokenizer stdin */
    void *tok_stdout;  /* FILE* — read end of tokenizer stdout */
    int tok_pid;
    bool tok_ready;

    /* ── Runtime context (forward pass activations) ── */
    fwd_ctx *fwd;
    uint32_t vocab_size;

    /* ── Config ── */
    float temperature;
    int32_t max_new_tokens;
} glm_engine;

const char *glm_tensor_error(void);

/* ── Lifecycle ── */
bool glm_engine_open(glm_engine *e, const char *model_path);
void glm_engine_close(glm_engine *e);
void glm_engine_print_summary(const glm_engine *e, FILE *fp);
const char *glm_engine_model_id(const glm_engine *e);

/* ── Tokenizer ── */
bool glm_engine_tokenize(glm_engine *e, const char *text, int32_t **out_ids, int32_t *out_n);
bool glm_engine_detokenize(glm_engine *e, const int32_t *ids, int32_t n, char **out_text);
bool glm_engine_encode_messages(glm_engine *e, const char *messages_json,
                                 int32_t **out_ids, int32_t *out_n);

/* ── Generation ── */
bool glm_engine_generation_available(const glm_engine *e);
int glm_engine_generate_text(glm_engine *e, const char *prompt, char *out, size_t out_cap);
int glm_engine_generate_stream(glm_engine *e, const char *prompt,
                                void (*callback)(const char *tok, void *arg), void *arg);

#endif