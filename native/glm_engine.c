#include "glm_engine.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(glm_engine *e, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->error, sizeof(e->error), fmt, ap);
    va_end(ap);
}

static bool contains_ci(const char *s, const char *needle) {
    if (!s || !needle) return false;
    const size_t nl = strlen(needle);
    for (; *s; s++) {
        size_t i = 0;
        while (i < nl && s[i] && tolower((unsigned char)s[i]) == tolower((unsigned char)needle[i])) i++;
        if (i == nl) return true;
    }
    return false;
}

static bool get_arch_u32(const glm_gguf_model *m, const char *arch, const char *suffix, uint32_t *out) {
    if (!arch) return false;
    char key[256];
    snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    return glm_gguf_get_u32(m, key, out);
}

static bool get_arch_u64(const glm_gguf_model *m, const char *arch, const char *suffix, uint64_t *out) {
    if (!arch) return false;
    char key[256];
    snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    return glm_gguf_get_u64(m, key, out);
}

static uint64_t count_tensor_name_contains(const glm_gguf_model *m, const char *needle) {
    uint64_t n = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (contains_ci(m->tensors[i].name, needle)) n++;
    }
    return n;
}

static void fill_info(glm_engine *e) {
    glm_model_info *info = &e->info;
    glm_gguf_model *m = &e->gguf;

    glm_gguf_get_string(m, "general.architecture", &info->architecture);
    glm_gguf_get_string(m, "general.name", &info->name);
    glm_gguf_get_string(m, "tokenizer.chat_template", &info->chat_template);

    const char *arch = info->architecture;
    get_arch_u32(m, arch, "block_count", &info->block_count);
    get_arch_u64(m, arch, "context_length", &info->context_length);
    get_arch_u64(m, arch, "max_position_embeddings", &info->context_length);
    get_arch_u32(m, arch, "embedding_length", &info->embedding_length);
    get_arch_u32(m, arch, "feed_forward_length", &info->feed_forward_length);
    get_arch_u32(m, arch, "attention.head_count", &info->attention_head_count);
    get_arch_u32(m, arch, "attention.head_count_kv", &info->attention_head_count_kv);
    get_arch_u32(m, arch, "expert_count", &info->expert_count);
    get_arch_u32(m, arch, "expert_used_count", &info->expert_used_count);

    info->expert_tensor_count = count_tensor_name_contains(m, "expert");
    info->attention_tensor_count = count_tensor_name_contains(m, "attn");
    info->block_tensor_count = count_tensor_name_contains(m, "blk.");
}

bool glm_engine_open(glm_engine *e, const char *model_path) {
    memset(e, 0, sizeof(*e));
    if (!model_path || !model_path[0]) {
        set_error(e, "missing model path");
        return false;
    }
    snprintf(e->model_path, sizeof(e->model_path), "%s", model_path);

    if (!glm_gguf_open(&e->gguf, model_path)) {
        set_error(e, "%s", e->gguf.error[0] ? e->gguf.error : "failed to open GGUF");
        return false;
    }

    fill_info(e);
    if (!e->info.architecture || !contains_ci(e->info.architecture, "glm")) {
        set_error(e, "GGUF architecture is not GLM-like; got %s",
                  e->info.architecture ? e->info.architecture : "(missing)");
        return false;
    }
    return true;
}

void glm_engine_close(glm_engine *e) {
    if (!e) return;
    free(e->info.architecture);
    free(e->info.name);
    free(e->info.chat_template);
    glm_gguf_close(&e->gguf);
    memset(e, 0, sizeof(*e));
}

void glm_engine_print_summary(const glm_engine *e, FILE *fp) {
    fprintf(fp, "glm-native engine\n");
    fprintf(fp, "model path:          %s\n", e->model_path);
    fprintf(fp, "model id:            %s\n", glm_engine_model_id(e));
    fprintf(fp, "architecture:        %s\n", e->info.architecture ? e->info.architecture : "(missing)");
    fprintf(fp, "GGUF version:        %u\n", e->gguf.version);
    fprintf(fp, "metadata entries:    %" PRIu64 "\n", e->gguf.n_kv);
    fprintf(fp, "tensor count:        %" PRIu64 "\n", e->gguf.n_tensors);
    fprintf(fp, "layers:              %u\n", e->info.block_count);
    fprintf(fp, "context length:      %" PRIu64 "\n", e->info.context_length);
    fprintf(fp, "embedding length:    %u\n", e->info.embedding_length);
    fprintf(fp, "attention heads:     %u\n", e->info.attention_head_count);
    fprintf(fp, "kv heads:            %u\n", e->info.attention_head_count_kv);
    fprintf(fp, "experts:             %u\n", e->info.expert_count);
    fprintf(fp, "experts used:        %u\n", e->info.expert_used_count);
    fprintf(fp, "expert tensors:      %" PRIu64 "\n", e->info.expert_tensor_count);
    fprintf(fp, "attention tensors:   %" PRIu64 "\n", e->info.attention_tensor_count);
    fprintf(fp, "block tensors:       %" PRIu64 "\n", e->info.block_tensor_count);
    fprintf(fp, "chat template:       %s\n", e->info.chat_template ? "yes" : "missing");
    fprintf(fp, "generation:          not implemented yet\n");
}

const char *glm_engine_model_id(const glm_engine *e) {
    return e->info.name && e->info.name[0] ? e->info.name : "glm-4.7-flash-native";
}

bool glm_engine_generation_available(const glm_engine *e) {
    (void)e;
    return false;
}

int glm_engine_generate_text(glm_engine *e, const char *prompt, char *out, size_t out_cap) {
    (void)e;
    (void)prompt;
    if (out_cap) {
        snprintf(out, out_cap,
                 "GLM native generation is not implemented yet. "
                 "The custom engine currently loads and validates the GGUF model only.");
    }
    return -1;
}
