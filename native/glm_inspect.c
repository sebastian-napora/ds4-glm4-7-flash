#include "glm_gguf.h"
#include "glm_native.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void print_usage(const char *argv0) {
    fprintf(stderr, "Usage: %s [--metadata] [--tensors] MODEL.gguf\n", argv0);
}

static void print_metadata(const glm_gguf_model *m) {
    puts("\nmetadata:");
    for (uint64_t i = 0; i < m->n_kv; i++) {
        printf("  %-56s %s\n", m->kv[i].key, glm_gguf_value_type_name(m->kv[i].type));
    }
}

static void print_tensors(const glm_gguf_model *m) {
    puts("\ntensors:");
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const glm_gguf_tensor *t = &m->tensors[i];
        printf("  %-64s %-10s [", t->name, glm_gguf_tensor_type_name(t->type));
        for (uint32_t d = 0; d < t->ndim; d++) {
            printf("%s%" PRIu64, d ? " x " : "", t->dim[d]);
        }
        printf("]\n");
    }
}

static bool read_arch(const glm_gguf_model *m, char **arch) {
    if (glm_gguf_get_string(m, "general.architecture", arch)) return true;
    *arch = NULL;
    return false;
}

static bool get_arch_u32(const glm_gguf_model *m, const char *arch, const char *suffix, uint32_t *out) {
    char key[256];
    snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    return glm_gguf_get_u32(m, key, out);
}

static bool get_arch_u64(const glm_gguf_model *m, const char *arch, const char *suffix, uint64_t *out) {
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

static int validate_glm(const glm_gguf_model *m) {
    int warnings = 0;
    char *arch = NULL;

    puts("\nGLM native validation:");
    if (!read_arch(m, &arch)) {
        puts("  WARN missing general.architecture");
        warnings++;
    } else {
        printf("  architecture:      %s\n", arch);
        if (!contains_ci(arch, "glm")) {
            puts("  WARN architecture does not look GLM-like");
            warnings++;
        }
    }

    if (arch) {
        uint32_t u32 = 0;
        uint64_t u64 = 0;
        if (get_arch_u32(m, arch, "block_count", &u32)) {
            printf("  layers:            %u\n", u32);
            if (u32 != 47) puts("  NOTE layer count differs from GLM-4.7-Flash public config");
        } else {
            puts("  WARN missing block_count metadata");
            warnings++;
        }

        if (get_arch_u64(m, arch, "context_length", &u64) || get_arch_u64(m, arch, "max_position_embeddings", &u64)) {
            printf("  context length:    %" PRIu64 "\n", u64);
        } else {
            puts("  WARN missing context length metadata");
            warnings++;
        }

        if (get_arch_u32(m, arch, "expert_count", &u32) ||
            get_arch_u32(m, arch, "expert_used_count", &u32) ||
            get_arch_u32(m, arch, "expert_feed_forward_length", &u32)) {
            puts("  MoE metadata:      present");
        } else {
            puts("  NOTE no obvious llama.cpp MoE metadata key found; inspect --metadata output");
        }
    }

    char *name = NULL;
    if (glm_gguf_get_string(m, "general.name", &name)) {
        printf("  name:              %s\n", name);
        free(name);
    }

    char *chat_template = NULL;
    if (glm_gguf_get_string(m, "tokenizer.chat_template", &chat_template)) {
        printf("  chat template:     yes (%zu bytes)\n", strlen(chat_template));
        free(chat_template);
    } else {
        puts("  NOTE no tokenizer.chat_template metadata found");
    }

    printf("  tensor names with expert: %" PRIu64 "\n", count_tensor_name_contains(m, "expert"));
    printf("  tensor names with attn:   %" PRIu64 "\n", count_tensor_name_contains(m, "attn"));
    printf("  tensor names with blk.:   %" PRIu64 "\n", count_tensor_name_contains(m, "blk."));

    free(arch);
    return warnings;
}

int main(int argc, char **argv) {
    bool show_metadata = false;
    bool show_tensors = false;
    const char *path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--metadata") == 0) {
            show_metadata = true;
        } else if (strcmp(argv[i], "--tensors") == 0) {
            show_tensors = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (!path) {
            path = argv[i];
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    if (!path) {
        print_usage(argv[0]);
        return 2;
    }

    glm_gguf_model model;
    if (!glm_gguf_open(&model, path)) {
        fprintf(stderr, "glm-inspect: %s\n", model.error[0] ? model.error : "failed to read GGUF");
        glm_gguf_close(&model);
        return 1;
    }

    printf("glm-native %s\n", GLM_NATIVE_VERSION);
    printf("file: %s\n", path);
    glm_gguf_print_summary(&model, stdout);

    int warnings = validate_glm(&model);
    if (show_metadata) print_metadata(&model);
    if (show_tensors) print_tensors(&model);

    glm_gguf_close(&model);
    return warnings ? 3 : 0;
}
