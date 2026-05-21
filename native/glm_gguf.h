#ifndef GLM_GGUF_H
#define GLM_GGUF_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define GLM_GGUF_MAGIC 0x46554747u
#define GLM_GGUF_MAX_DIMS 8

typedef enum {
    GLM_GGUF_VALUE_UINT8 = 0,
    GLM_GGUF_VALUE_INT8 = 1,
    GLM_GGUF_VALUE_UINT16 = 2,
    GLM_GGUF_VALUE_INT16 = 3,
    GLM_GGUF_VALUE_UINT32 = 4,
    GLM_GGUF_VALUE_INT32 = 5,
    GLM_GGUF_VALUE_FLOAT32 = 6,
    GLM_GGUF_VALUE_BOOL = 7,
    GLM_GGUF_VALUE_STRING = 8,
    GLM_GGUF_VALUE_ARRAY = 9,
    GLM_GGUF_VALUE_UINT64 = 10,
    GLM_GGUF_VALUE_INT64 = 11,
    GLM_GGUF_VALUE_FLOAT64 = 12,
} glm_gguf_value_type;

typedef struct {
    char *key;
    uint32_t type;
    uint64_t value_pos;
} glm_gguf_kv;

typedef struct {
    char *name;
    uint32_t ndim;
    uint64_t dim[GLM_GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
} glm_gguf_tensor;

typedef struct {
    FILE *fp;
    char error[256];

    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint32_t alignment;
    uint64_t tensor_data_offset;

    glm_gguf_kv *kv;
    glm_gguf_tensor *tensors;
} glm_gguf_model;

bool glm_gguf_open(glm_gguf_model *m, const char *path);
void glm_gguf_close(glm_gguf_model *m);

const glm_gguf_kv *glm_gguf_find_kv(const glm_gguf_model *m, const char *key);
bool glm_gguf_get_string(const glm_gguf_model *m, const char *key, char **out);
bool glm_gguf_get_u32(const glm_gguf_model *m, const char *key, uint32_t *out);
bool glm_gguf_get_u64(const glm_gguf_model *m, const char *key, uint64_t *out);
bool glm_gguf_get_f32(const glm_gguf_model *m, const char *key, float *out);
bool glm_gguf_get_bool(const glm_gguf_model *m, const char *key, bool *out);

const char *glm_gguf_value_type_name(uint32_t type);
const char *glm_gguf_tensor_type_name(uint32_t type);
uint64_t glm_gguf_tensor_elems(const glm_gguf_tensor *t);
void glm_gguf_print_summary(const glm_gguf_model *m, FILE *fp);

#endif
