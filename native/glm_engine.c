#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "glm_engine.h"

// ─── Helpers ──────────────────────────────────────────────────────────────────
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
    get_arch_u32(m, arch, "block_count",        &info->block_count);
    get_arch_u64(m, arch, "context_length",       &info->context_length);
    get_arch_u64(m, arch, "max_position_embeddings", &info->context_length);
    get_arch_u32(m, arch, "embedding_length",      &info->embedding_length);
    get_arch_u32(m, arch, "feed_forward_length",  &info->feed_forward_length);
    get_arch_u32(m, arch, "attention.head_count", &info->attention_head_count);
    get_arch_u32(m, arch, "attention.head_count_kv", &info->attention_head_count_kv);
    get_arch_u32(m, arch, "expert_count",         &info->expert_count);
    get_arch_u32(m, arch, "expert_used_count",    &info->expert_used_count);

    info->expert_tensor_count   = count_tensor_name_contains(m, "expert");
    info->attention_tensor_count = count_tensor_name_contains(m, "attn");
    info->block_tensor_count    = count_tensor_name_contains(m, "blk.");
}

// ─── Tokenizer subprocess ──────────────────────────────────────────────────────
static bool tok_write(glm_engine *e, const char *req) {
    FILE *f = (FILE *)e->tok_stdin;
    if (!f || fprintf(f, "%s\n", req) < 0 || fflush(f) != 0) return false;
    return true;
}

static bool tok_read(glm_engine *e, char *out, size_t cap) {
    FILE *f = (FILE *)e->tok_stdout;
    if (!f) return false;
    if (!fgets(out, (int)cap, f)) return false;
    size_t n = strlen(out);
    if (n > 0 && out[n - 1] == '\n') out[n - 1] = 0;
    return true;
}

static bool tok_query(glm_engine *e, const char *req, char *out, size_t cap) {
    if (!tok_write(e, req)) return false;
    if (!tok_read(e, out, cap)) return false;
    return true;
}

static bool start_tokenizer(glm_engine *e, const char *python_path) {
    fprintf(stderr, "start_tokenizer: python=%s\n", python_path);
    int pin[2], pout[2];
    if (pipe(pin) < 0 || pipe(pout) < 0) { fprintf(stderr, "start_tokenizer: pipe failed\n"); return false; }
    fprintf(stderr, "start_tokenizer: pipes created\n");

    pid_t pid = fork();
    if (pid < 0) {
        close(pin[0]); close(pin[1]); close(pout[0]); close(pout[1]);
        return false;
    }

    if (pid == 0) {
        // Child: redirect stdin=pout[0], stdout=pin[1]
        close(pin[0]); close(pout[1]);
        dup2(pout[0], STDIN_FILENO);
        dup2(pin[1], STDOUT_FILENO);
        close(pout[0]); close(pin[1]);
        // Set PYTHONPATH so system python3 can find transformers in venv
        setenv("PYTHONPATH",
               "/home/sna/ai-projects/ds4-glm4-7-flash/venv/lib/python3.12/site-packages", 1);
        // Use system python3 directly (venv/python is a symlink to it)
        execl("/usr/bin/python3", "python3",
              (char *)"-u",
              (char *)"/home/sna/ai-projects/ds4-glm4-7-flash/native/glm_tokenize.py",
              (char *)NULL);
        _exit(127);
    }

    // Parent
    fprintf(stderr, "start_tokenizer: forked pid=%d\n", (int)pid);
    close(pin[1]); close(pout[0]);
    e->tok_stdin  = fdopen(pin[1], "w");
    e->tok_stdout = fdopen(pout[0], "r");
    e->tok_pid    = pid;
    e->tok_ready  = false;

    if (!e->tok_stdin || !e->tok_stdout) {
        fprintf(stderr, "start_tokenizer: fdopen failed stdin=%p stdout=%p\n", (void*)e->tok_stdin, (void*)e->tok_stdout);
        if (e->tok_stdin) fclose(e->tok_stdin);
        if (e->tok_stdout) fclose(e->tok_stdout);
        kill(pid, SIGTERM);
        waitpid(pid, NULL, 0);
        return false;
    }

    setvbuf(e->tok_stdin, NULL, _IONBF, 0);
    setvbuf(e->tok_stdout, NULL, _IONBF, 0);
    fprintf(stderr, "start_tokenizer: fdopen ok, sending vocab_info\n");

    // Ping with vocab_info to confirm it's up
    char buf[256];
    fprintf(stderr, "start_tokenizer: tok_query...\n");
    if (tok_query(e, "{\"type\":\"vocab_info\"}", buf, sizeof(buf))) {
        fprintf(stderr, "start_tokenizer: tok_query resp=%s\n", buf);
        if (contains_ci(buf, "\"vocab_size\"")) {
            e->tok_ready = true;
        }
    } else {
        fprintf(stderr, "start_tokenizer: tok_query failed\n");
    }

    if (!e->tok_ready) {
        fprintf(stderr, "glm_engine: tokenizer subprocess failed to respond\n");
    } else {
        fprintf(stderr, "glm_engine: tokenizer ready (pid=%d)\n", (int)e->tok_pid);
    }
    return e->tok_ready;
}

static void stop_tokenizer(glm_engine *e) {
    if (e->tok_pid <= 0) return;

    // Send termination signal
    if (e->tok_stdin) {
        fclose((FILE *)e->tok_stdin);
        e->tok_stdin = NULL;
    }
    int status;
    waitpid(e->tok_pid, &status, WNOHANG);
    e->tok_pid = 0;
    e->tok_ready = false;
}

bool glm_engine_tokenize(glm_engine *e, const char *text, int32_t **out_ids, int32_t *out_n) {
    if (!e->tok_ready) { set_error(e, "tokenizer not ready"); return false; }

    char req[512];
    snprintf(req, sizeof(req),
             "{\"type\":\"encode\",\"text\":%s}",
             text ? text : "");
    char resp[8192];
    if (!tok_query(e, req, resp, sizeof(resp))) {
        set_error(e, "tokenizer encode failed"); return false;
    }

    // Parse: {"type":"encode","ids":[...]}
    // Find '[' bracket
    const char *p = strchr(resp, '[');
    if (!p) { set_error(e, "bad tokenizer response: no '['"); return false; }

    // Count commas to determine count
    int32_t count = 0;
    for (const char *q = p; q && *q != ']'; q++) if (*q == ',') count++;
    count++; // last element

    int32_t *ids = (int32_t *)malloc((size_t)count * sizeof(int32_t));
    if (!ids) { set_error(e, "OOM"); return false; }

    const char *r = p;
    for (int32_t i = 0; i < count; i++) {
        while (r && *r && (*r < '0' || *r > '9') && *r != '-') r++;
        if (!r || !*r || *r == ']') { ids[i] = 0; break; }
        char *end;
        long v = strtol(r, &end, 10);
        ids[i] = (int32_t)v;
        r = end;
    }

    *out_ids = ids;
    *out_n = count;
    return true;
}

bool glm_engine_detokenize(glm_engine *e, const int32_t *ids, int32_t n, char **out_text) {
    if (!e->tok_ready) { set_error(e, "tokenizer not ready"); return false; }
    if (!ids || n <= 0) { set_error(e, "invalid ids/n"); return false; }

    // Build JSON array string
    char ids_str[8192];
    int pos = 0;
    int rem = (int)sizeof(ids_str);
    ids_str[pos++] = '[';
    for (int32_t i = 0; i < n && rem > 8; i++) {
        if (i > 0) { ids_str[pos++] = ','; rem--; }
        int w = snprintf(ids_str + pos, (size_t)rem, "%d", ids[i]);
        if (w < 0) break;
        pos += w; rem -= w;
    }
    ids_str[pos++] = ']';
    ids_str[pos++] = 0;

    char req[10240];
    snprintf(req, sizeof(req), "{\"type\":\"decode\",\"ids\":%s}", ids_str);
    char resp[8192];
    if (!tok_query(e, req, resp, sizeof(resp))) {
        set_error(e, "tokenizer decode failed"); return false;
    }

    // Parse: {"type":"decode","text":"..."}
    const char *p = strstr(resp, "\"text\"");
    if (!p) { set_error(e, "bad decode response"); return false; }
    p = strchr(p, ':');
    if (!p) { set_error(e, "bad decode response 2"); return false; }
    p++;
    while (*p && (*p <= ' ' || *p == '\"')) p++;
    // Trim trailing }
    char *end = ids_str;
    strncpy(end, p, 256 - 1);
    end[255] = 0;
    int len = (int)strlen(end);
    while (len > 0 && (end[len - 1] == '"' || end[len - 1] == '}')) len--;
    end[len] = 0;

    char *out = (char *)malloc((size_t)len + 1);
    if (!out) { set_error(e, "OOM"); return false; }
    memcpy(out, end, (size_t)len);
    out[len] = 0;
    *out_text = out;
    return true;
}

bool glm_engine_encode_messages(glm_engine *e, const char *messages_json,
                                 int32_t **out_ids, int32_t *out_n) {
    if (!e->tok_ready) { set_error(e, "tokenizer not ready"); return false; }

    char req[8192];
    snprintf(req, sizeof(req),
             "{\"type\":\"encode_messages\",\"messages\":%s}",
             messages_json && messages_json[0] ? messages_json : "[]");
    char resp[8192];
    if (!tok_query(e, req, resp, sizeof(resp))) {
        set_error(e, "tokenizer encode_messages failed"); return false;
    }

    // Parse JSON: find "ids":[...] part
    const char *p = strstr(resp, "\"ids\"");
    if (!p) { set_error(e, "bad encode_messages response"); return false; }
    p = strchr(p, '[');
    if (!p) { set_error(e, "no '[' in ids"); return false; }

    int32_t count = 0;
    for (const char *q = p; q && *q != ']'; q++) if (*q == ',') count++;
    count++;

    int32_t *ids = (int32_t *)malloc((size_t)count * sizeof(int32_t));
    if (!ids) { set_error(e, "OOM"); return false; }

    const char *r = p;
    for (int32_t i = 0; i < count; i++) {
        while (r && *r && (*r < '0' || *r > '9') && *r != '-') r++;
        if (!r || !*r || *r == ']') { ids[i] = 0; break; }
        char *end;
        long v = strtol(r, &end, 10);
        ids[i] = (int32_t)v;
        r = end;
    }

    *out_ids = ids;
    *out_n = count;
    return true;
}

// ─── Engine lifecycle ──────────────────────────────────────────────────────────
bool glm_engine_open(glm_engine *e, const char *model_path) {
    memset(e, 0, sizeof(*e));
    e->tok_pid = -1;
    e->weights_loaded = false;
    e->fwd = NULL;
    e->temperature = 0.0f;
    e->max_new_tokens = 256;

    if (!model_path || !model_path[0]) {
        set_error(e, "missing model path"); return false;
    }
    snprintf(e->model_path, sizeof(e->model_path), "%s", model_path);

    if (!glm_gguf_open(&e->gguf, model_path)) {
        set_error(e, "%s", e->gguf.error[0] ? e->gguf.error : "failed to open GGUF");
        return false;
    }

    fill_info(e);

    if (!e->info.architecture) {
        set_error(e, "GGUF architecture is missing");
        return false;
    }
    if (!contains_ci(e->info.architecture, "glm") &&
        !contains_ci(e->info.architecture, "deepseek")) {
        set_error(e, "unsupported architecture: %s (expected GLM or DeepSeek family)",
                  e->info.architecture);
        return false;
    }

    fprintf(stderr, "\nLoading model weights into memory...\n");
    if (!glm_weights_load(&e->weights, &e->gguf)) {
        set_error(e, "failed to load weights: %s", glm_tensor_error());
        return false;
    }
    e->weights_loaded = true;
    fprintf(stderr, "\n");

    // Init forward context (max 2048 tokens, one batch)
    e->fwd = (fwd_ctx *)calloc(1, sizeof(fwd_ctx));
    if (!e->fwd || !fwd_ctx_init(e->fwd, 2048)) {
        set_error(e, "failed to init forward context (OOM?)");
        return false;
    }

    // Start tokenizer subprocess — try venv python first, then system python3
    const char *py = getenv("GLM_PYTHON");
    if (!py || !py[0]) {
        char *cwd = getenv("PWD");
        // Prefer venv python from project directory
        static char venv_python[512];
        if (cwd) {
            snprintf(venv_python, sizeof(venv_python), "%s/venv/bin/python", cwd);
            if (access(venv_python, X_OK) == 0) py = venv_python;
        }
        if (!py) py = "python3";
    }
    if (!start_tokenizer(e, py)) {
        fprintf(stderr, "glm_engine: tokenizer unavailable — generation will use raw token IDs\n");
    } else {
        fprintf(stderr, "glm_engine: tokenizer ready (pid=%d)\n", (int)e->tok_pid);
    }

    // Attempt vocab detection from tokenizer
    if (e->tok_ready) {
        char resp[256];
        if (tok_query(e, "{\"type\":\"vocab_info\"}", resp, sizeof(resp))) {
            // Extract vocab_size
            const char *p = strstr(resp, "\"vocab_size\"");
            if (p) {
                long vs = strtol(p + 12, NULL, 10);
                if (vs > 0) e->vocab_size = (uint32_t)vs;
            }
        }
        if (!e->vocab_size) e->vocab_size = FWD_VOCAB;
    } else {
        e->vocab_size = FWD_VOCAB;
    }

    return true;
}

void glm_engine_close(glm_engine *e) {
    if (!e) return;
    stop_tokenizer(e);
    if (e->tok_stdout) { fclose((FILE *)e->tok_stdout); e->tok_stdout = NULL; }
    if (e->fwd) { fwd_ctx_free(e->fwd); free(e->fwd); e->fwd = NULL; }
    if (e->weights_loaded) { glm_weights_free(&e->weights); e->weights_loaded = false; }
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
    fprintf(fp, "vocab size:          %u\n", e->vocab_size);
    fprintf(fp, "embedding length:    %u\n", e->info.embedding_length);
    fprintf(fp, "attention heads:     %u\n", e->info.attention_head_count);
    fprintf(fp, "kv heads:            %u\n", e->info.attention_head_count_kv);
    fprintf(fp, "experts:             %u\n", e->info.expert_count);
    fprintf(fp, "experts used:        %u\n", e->info.expert_used_count);
    fprintf(fp, "expert tensors:      %" PRIu64 "\n", e->info.expert_tensor_count);
    fprintf(fp, "attention tensors:   %" PRIu64 "\n", e->info.attention_tensor_count);
    fprintf(fp, "block tensors:       %" PRIu64 "\n", e->info.block_tensor_count);
    fprintf(fp, "chat template:       %s\n", e->info.chat_template ? "yes" : "missing");
    fprintf(fp, "tokenizer:           %s\n", e->tok_ready ? "ready (HF subprocess)" : "not available");
    fprintf(fp, "generation:          %s\n",
            (e->fwd && e->weights_loaded) ? "enabled" : "disabled");
}

const char *glm_engine_model_id(const glm_engine *e) {
    return e->info.name && e->info.name[0] ? e->info.name : "glm-4.7-flash-native";
}

// ─── Generation ────────────────────────────────────────────────────────────────
bool glm_engine_generation_available(const glm_engine *e) {
    return e && e->fwd && e->weights_loaded;
}

int glm_engine_generate_text(glm_engine *e, const char *prompt, char *out, size_t out_cap) {
    if (!e || !e->fwd || !e->weights_loaded) {
        if (out_cap) snprintf(out, out_cap, "glm_engine not ready");
        return -1;
    }

    if (!prompt || !prompt[0]) {
        if (out_cap) out[0] = 0;
        return 0;
    }

    // Tokenize the prompt
    int32_t *input_ids = NULL;
    int32_t n_input = 0;
    if (!glm_engine_tokenize(e, prompt, &input_ids, &n_input)) {
        char msg[512];
        snprintf(msg, sizeof(msg), "tokenization failed: %.200s", e->error);
        snprintf(e->error, sizeof(e->error), "%s", msg);
        if (out_cap) snprintf(out, out_cap, "%s", msg);
        return -1;
    }

    if (n_input <= 0 || n_input > e->fwd->max_seq) {
        free(input_ids);
        if (out_cap) snprintf(out, out_cap, "bad token count: %d", n_input);
        return -1;
    }

    // Generate tokens
    int32_t output_tokens[512];
    int32_t n_gen = glm_generate(e->fwd, &e->weights,
                                 input_ids, n_input,
                                 output_tokens,
                                 e->max_new_tokens,
                                 e->temperature);

    free(input_ids);

    if (n_gen <= 0) {
        if (out_cap) snprintf(out, out_cap, "generation produced no tokens");
        return -1;
    }

    // Detokenize output
    char *decoded = NULL;
    if (glm_engine_detokenize(e, output_tokens, n_gen, &decoded)) {
        if (out_cap) strncpy(out, decoded, out_cap - 1);
        out[out_cap - 1] = 0;
        free(decoded);
    } else {
        // Fallback: just format as comma-separated token IDs
        if (out_cap > 0) {
            int pos = 0;
            for (int32_t i = 0; i < n_gen && pos < (int)out_cap - 20; i++) {
                int w = snprintf(out + pos, out_cap - (size_t)pos, "%d ", output_tokens[i]);
                if (w > 0) pos += w;
            }
        }
    }

    return n_gen;
}

int glm_engine_generate_stream(glm_engine *e, const char *prompt,
                                void (*callback)(const char *tok, void *arg), void *arg) {
    if (!e || !e->fwd || !e->weights_loaded || !callback) return -1;

    int32_t *input_ids = NULL;
    int32_t n_input = 0;
    if (!glm_engine_tokenize(e, prompt, &input_ids, &n_input)) return -1;

    if (n_input <= 0 || n_input > e->fwd->max_seq) {
        free(input_ids);
        return -1;
    }

    int32_t output_tokens[512];
    int32_t n_gen = glm_generate(e->fwd, &e->weights,
                                 input_ids, n_input,
                                 output_tokens,
                                 e->max_new_tokens,
                                 e->temperature);

    free(input_ids);

    if (n_gen <= 0) return 0;

    // Stream one token at a time through detokenizer
    for (int32_t i = 0; i < n_gen; i++) {
        char *tok_str = NULL;
        if (glm_engine_detokenize(e, &output_tokens[i], 1, &tok_str)) {
            callback(tok_str, arg);
            free(tok_str);
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "[%d]", output_tokens[i]);
            callback(buf, arg);
        }
    }

    return n_gen;
}