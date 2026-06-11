#include "glm_engine.h"
#include "glm_native.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define REQ_MAX  65536
#define ID_MAX    256
#define MODEL_MAX  256

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

// ─── String helpers ───────────────────────────────────────────────────────────
static void json_escape(const char *src, char *dst, size_t cap) {
    size_t j = 0;
    for (const char *p = src; *p && j + 4 < cap; p++) {
        switch (*p) {
            case '"':  if (j + 2 < cap) { dst[j++] = '\\'; dst[j++] = '"';  } break;
            case '\\': if (j + 2 < cap) { dst[j++] = '\\'; dst[j++] = '\\'; } break;
            case '\n': if (j + 2 < cap) { dst[j++] = '\\'; dst[j++] = 'n';  } break;
            case '\r': if (j + 2 < cap) { dst[j++] = '\\'; dst[j++] = 'r';  } break;
            case '\t': if (j + 2 < cap) { dst[j++] = '\\'; dst[j++] = 't';  } break;
            default:   dst[j++] = *p; break;
        }
    }
    dst[j] = 0;
}

// Find "messages" JSON array inside the body
static const char *skip_ws(const char *p) {
    while (*p && (*p <= ' ' || *p == ':' || *p == ',')) p++;
    return p;
}

static const char *skip_string(const char *p) {
    if (*p == '"') p++;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) p++;
        p++;
    }
    if (*p == '"') p++;
    return p;
}

// Extract role from {"role":"user",...} object
static bool extract_role(const char *obj_start, char *role_buf, size_t role_cap) {
    const char *p = obj_start;
    role_buf[0] = 0;
    const char *key = strstr(p, "\"role\"");
    if (!key || key > p + 256) return false;
    const char *colon = strchr(key, ':');
    if (!colon) return false;
    const char *val = skip_ws(colon + 1);
    if (*val != '"') return false;
    val++;
    size_t i = 0;
    while (*val && *val != '"' && i + 1 < role_cap) role_buf[i++] = *val++;
    role_buf[i] = 0;
    return i > 0;
}

// Extract content string from object
static bool extract_content(const char *obj_start, char *content_buf, size_t content_cap) {
    const char *p = obj_start;
    const char *key = strstr(p, "\"content\"");
    if (!key || key > p + 2048) return false;
    const char *colon = strchr(key, ':');
    if (!colon) return false;
    const char *val = skip_ws(colon + 1);
    if (*val != '"') {
        // Content might be an array (tool calls etc) — just find next string
        val = strchr(val, '"');
        if (!val) return false;
    }
    val++; // skip opening "
    size_t i = 0;
    while (*val && *val != '"' && i + 1 < content_cap) {
        if (*val == '\\' && val[1]) { val++; }
        content_buf[i++] = *val++;
    }
    content_buf[i] = 0;
    return i > 0;
}

// Build a simple JSON array of messages for the tokenizer
// Converts [{"role":"user","content":"hi"},{"role":"assistant","content":"hi back"}]
// into JSON array string with escaped content
static char *build_messages_json(const char *body) {
    // Find "messages":[
    const char *mp = strstr(body, "\"messages\"");
    if (!mp) return NULL;
    const char *arr = strchr(mp, '[');
    if (!arr) return NULL;

    // Count depth to find matching ]
    int depth = 0;
    const char *end = arr;
    const char *p = arr;
    while (*p) {
        if (*p == '[' || *p == '{') depth++;
        else if (*p == ']' || *p == '}') { depth--; if (depth == 0) { end = p + 1; break; } }
        p++;
    }

    // Copy the array portion
    size_t len = (size_t)(end - arr);
    char *raw = (char *)malloc(len + 1);
    if (!raw) return NULL;
    memcpy(raw, arr, len);
    raw[len] = 0;

    // Build new array with cleaned content strings
    // Replace \" with escaped form for tokenizer, track messages
    char *result = (char *)malloc(len * 3 + 16);
    if (!result) { free(raw); return NULL; }

    char *r = result;
    r += sprintf(r, "[");
    size_t pos = 0;
    bool first_msg = true;

    while (pos < len) {
        while (pos < len && raw[pos] <= ' ') pos++;
        if (pos >= len || raw[pos] == ']') break;

        if (raw[pos] == '{') {
            // Find matching }
            int d = 1;
            size_t obj_start = pos;
            size_t obj_end = pos + 1;
            for (size_t i = pos + 1; i < len && d > 0; i++) {
                if (raw[i] == '{') d++;
                else if (raw[i] == '}') { d--; if (d == 0) obj_end = i + 1; }
            }

            char role_buf[32] = {0};
            char content_buf[4096] = {0};
            extract_role(raw + obj_start, role_buf, sizeof(role_buf));
            extract_content(raw + obj_start, content_buf, sizeof(content_buf));

            if (!first_msg) *r++ = ',';
            first_msg = false;

            // Escape content for JSON
            char esc_content[8192];
            json_escape(content_buf, esc_content, sizeof(esc_content));
            r += sprintf(r, "{\"role\":\"%s\",\"content\":\"%s\"}",
                         role_buf, esc_content);

            pos = obj_end;
        } else {
            pos++;
        }
    }

    r += sprintf(r, "]");
    free(raw);
    return result;
}

// ─── HTTP helpers ──────────────────────────────────────────────────────────────
static bool send_all(int fd, const char *buf, size_t len) {
    while (len) {
        ssize_t n = send(fd, buf, len, 0);
        if (n <= 0) return false;
        buf += n;
        len -= (size_t)n;
    }
    return true;
}

static void http_response(int fd, int status, const char *reason,
                          const char *content_type, const char *body, size_t body_len) {
    char header[512];
    int n = snprintf(header, sizeof(header),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: %s\r\n"
                     "Content-Length: %zu\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "Connection: close\r\n"
                     "\r\n",
                     status, reason, content_type, body_len);
    send_all(fd, header, (size_t)n);
    if (body && body_len > 0) send_all(fd, body, body_len);
}

static void http_json(int fd, int status, const char *reason, const char *body) {
    http_response(fd, status, reason, "application/json", body, strlen(body));
}

static void http_error(int fd, int status, const char *message) {
    char body[1024];
    snprintf(body, sizeof(body),
             "{\"error\":{\"message\":\"%s\",\"type\":\"internal_error\",\"code\":\"glm_native\"}}\n",
             message);
    http_json(fd, status, "Error", body);
}

// SSE chunk
static void sse_chunk(int fd, const char *data) {
    char buf[8192];
    int n = snprintf(buf, sizeof(buf), "data: %s\n\n", data);
    send_all(fd, buf, (size_t)n);
}

static void sse_done(int fd) {
    send_all(fd, "data: [DONE]\n\n", 14);
}

// ─── Streaming callback ─────────────────────────────────────────────────────────
struct stream_ctx {
    int client_fd;
    const char *model_id;
    int token_count;
};

// Called once per generated token
static void stream_token_cb(const char *tok_str, void *arg) {
    struct stream_ctx *s = (struct stream_ctx *)arg;
    s->token_count++;

    // Build SSE data: {"choices":[{"delta":{"content":"token"}}]}
    char esc[512];
    json_escape(tok_str, esc, sizeof(esc));
    char chunk[1024];
    snprintf(chunk, sizeof(chunk),
             "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"}}]}",
             esc);
    sse_chunk(s->client_fd, chunk);
}

// ─── Route handlers ────────────────────────────────────────────────────────────
static void handle_health(int fd, glm_engine *engine) {
    (void)engine;
    char body[1024];
    snprintf(body, sizeof(body),
             "{\"status\":\"ok\",\"engine\":\"glm-native\",\"version\":\"%s\","
             "\"model\":\"%s\",\"generation\":%s}\n",
             GLM_NATIVE_VERSION,
             glm_engine_model_id(engine),
             glm_engine_generation_available(engine) ? "true" : "false");
    http_json(fd, 200, "OK", body);
}

static void handle_models(int fd, glm_engine *engine) {
    char body[2048];
    snprintf(body, sizeof(body),
             "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
             "\"owned_by\":\"local\",\"root\":\"%s\"}]}\n",
             glm_engine_model_id(engine), glm_engine_model_id(engine));
    http_json(fd, 200, "OK", body);
}

static void handle_chat_completions(int fd, const char *body, size_t body_len,
                                    glm_engine *engine) {
    // Parse streaming flag
    bool stream = false;
    const char *sp = strstr(body, "\"stream\"");
    if (sp) {
        const char *colon = strchr(sp, ':');
        if (colon) {
            const char *val = skip_ws(colon + 1);
            if (val[0] == 't' || val[0] == 'T' || val[0] == '1') stream = true;
        }
    }

    // Extract model name (default if not specified)
    char model_id[MODEL_MAX] = "glm-4.7-flash-native";
    const char *mp = strstr(body, "\"model\"");
    if (mp) {
        const char *colon = strchr(mp, ':');
        if (colon) {
            const char *val = skip_ws(colon + 1);
            if (*val == '"') {
                val++;
                size_t i = 0;
                while (*val && *val != '"' && i + 1 < sizeof(model_id)) {
                    model_id[i++] = *val++;
                }
                model_id[i] = 0;
            }
        }
    }

    // Extract max_tokens
    int max_tokens = 256;
    const char *tp = strstr(body, "\"max_tokens\"");
    if (tp) {
        const char *colon = strchr(tp, ':');
        if (colon) {
            long v = strtol(skip_ws(colon + 1), NULL, 10);
            if (v > 0 && v < 2048) max_tokens = (int)v;
        }
    }

    // Extract temperature
    float temperature = 0.0f;
    const char *tempp = strstr(body, "\"temperature\"");
    if (tempp) {
        const char *colon = strchr(tempp, ':');
        if (colon) {
            double v = strtod(skip_ws(colon + 1), NULL);
            if (v >= 0.0 && v <= 2.0) temperature = (float)v;
        }
    }

    // Build messages JSON for tokenizer
    char *msgs_json = build_messages_json(body);
    if (!msgs_json) {
        http_error(fd, 400, "missing or invalid messages array");
        return;
    }

    // Encode messages using tokenizer subprocess
    int32_t *input_ids = NULL;
    int32_t n_input = 0;
    if (!glm_engine_encode_messages(engine, msgs_json, &input_ids, &n_input)) {
        free(msgs_json);
        http_error(fd, 500, "tokenization failed");
        return;
    }
    free(msgs_json);

    if (n_input <= 0 || n_input > engine->fwd->max_seq) {
        free(input_ids);
        http_error(fd, 400, "invalid token count");
        return;
    }

    if (stream) {
        // SSE stream response
        char header[512];
        int hn = snprintf(header, sizeof(header),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n");
        send_all(fd, header, (size_t)hn);

        // Send role info
        sse_chunk(fd, "{\"choices\":[{\"index\":0,\"delta\":{}}]}");

        // Generate
        struct stream_ctx sctx = { .client_fd = fd, .model_id = model_id, .token_count = 0 };
        engine->max_new_tokens = max_tokens;
        engine->temperature = temperature;

        int32_t output_tokens[2048];
        int32_t n_gen = glm_generate(engine->fwd, &engine->weights,
                                      input_ids, n_input,
                                      output_tokens, max_tokens, temperature);

        // Stream tokens through detokenizer
        for (int32_t i = 0; i < n_gen; i++) {
            char *tok_str = NULL;
            if (glm_engine_detokenize(engine, &output_tokens[i], 1, &tok_str)) {
                stream_token_cb(tok_str, &sctx);
                free(tok_str);
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "[tok:%d]", output_tokens[i]);
                sse_chunk(fd, buf);
            }
        }

        free(input_ids);
        sse_done(fd);
        return;
    }

    // Non-streaming: collect all output then respond
    engine->max_new_tokens = max_tokens;
    engine->temperature = temperature;

    int32_t output_tokens[2048];
    int32_t n_gen = glm_generate(engine->fwd, &engine->weights,
                                  input_ids, n_input,
                                  output_tokens, max_tokens, temperature);
    free(input_ids);

    if (n_gen < 0) n_gen = 0;

    // Detokenize
    char full_text[8192] = {0};
    int full_pos = 0;

    for (int32_t i = 0; i < n_gen && full_pos < (int)sizeof(full_text) - 256; i++) {
        char *tok_str = NULL;
        if (glm_engine_detokenize(engine, &output_tokens[i], 1, &tok_str)) {
            size_t avail = sizeof(full_text) - (size_t)full_pos - 1;
            strncat(full_text + full_pos, tok_str, avail);
            full_pos += (int)strlen(tok_str);
            free(tok_str);
        }
    }

    // Build response JSON
    char esc_text[16384];
    json_escape(full_text, esc_text, sizeof(esc_text));

    char response[32768];
    int n = snprintf(response, sizeof(response),
        "{\n"
        "  \"id\":\"chatcmpl-%08x\",\n"
        "  \"object\":\"chat.completion\",\n"
        "  \"created\":%ld,\n"
        "  \"model\":\"%s\",\n"
        "  \"choices\":[{\n"
        "    \"index\":0,\n"
        "    \"message\":{\n"
        "      \"role\":\"assistant\",\n"
        "      \"content\":\"%s\"\n"
        "    },\n"
        "    \"finish_reason\":\"stop\"\n"
        "  }],\n"
        "  \"usage\":{\n"
        "    \"prompt_tokens\":%d,\n"
        "    \"completion_tokens\":%d,\n"
        "    \"total_tokens\":%d\n"
        "  }\n"
        "}\n",
        (unsigned)time(NULL) ^ (unsigned)(fd & 0xFFFF),
        (long)time(NULL),
        model_id,
        esc_text,
        n_input, n_gen, n_input + n_gen);

    http_json(fd, 200, "OK", response);
}

// ─── Main ──────────────────────────────────────────────────────────────────────
static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s -m MODEL.gguf [--host HOST] [--port PORT]\n"
            "\n"
            "Endpoints:\n"
            "  GET  /health\n"
            "  GET  /v1/models\n"
            "  POST /v1/chat/completions  (streaming and non-streaming)\n"
            "  POST /v1/completions\n",
            argv0);
}

static void handle_client(int fd, glm_engine *engine) {
    char req[REQ_MAX + 1];
    ssize_t n = recv(fd, req, REQ_MAX, 0);
    if (n <= 0) return;
    req[n] = 0;

    // Parse request line
    char method[16] = {0};
    char path[256] = {0};
    sscanf(req, "%15s %255s", method, path);

    // Find body (after \r\n\r\n)
    const char *body_start = strstr(req, "\r\n\r\n");
    if (!body_start) body_start = req + n;
    else body_start += 4;

    // CORS preflight — must precede any route check so OPTIONS works globally
    if (!strcmp(method, "OPTIONS")) {
        static const char opts[] =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With, *\r\n"
            "Access-Control-Max-Age: 86400\r\n"
            "Connection: close\r\n"
            "\r\n";
        send_all(fd, opts, sizeof(opts) - 1);
        return;
    }

    if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        handle_health(fd, engine);
        return;
    }
    if (!strcmp(method, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/v1/models/"))) {
        handle_models(fd, engine);
        return;
    }
    if (!strcmp(method, "POST") && (!strcmp(path, "/v1/chat/completions") || !strcmp(path, "/v1/completions"))) {
        handle_chat_completions(fd, body_start, (size_t)(req + n - body_start), engine);
        return;
    }

    http_json(fd, 404, "Not Found", "{\"error\":{\"message\":\"not found\"}}\n");
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *host = "0.0.0.0";
    int port = 11112;

    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) {
            model = argv[++i];
        } else if (!strcmp(argv[i], "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            port = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!model || port <= 0 || port > 65535) {
        usage(argv[0]);
        return 2;
    }

    glm_engine engine;
    if (!glm_engine_open(&engine, model)) {
        fprintf(stderr, "glm-native-server: %s\n", engine.error[0] ? engine.error : "failed to open model");
        glm_engine_close(&engine);
        return 1;
    }

    glm_engine_print_summary(&engine, stderr);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); glm_engine_close(&engine); return 1; }

    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind"); close(server_fd); glm_engine_close(&engine); return 1;
    }
    if (listen(server_fd, 16) != 0) {
        perror("listen"); close(server_fd); glm_engine_close(&engine); return 1;
    }

    fprintf(stderr, "glm-native-server listening on http://%s:%d\n", host, port);

    while (!g_stop) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        handle_client(client, &engine);
        close(client);
    }

    close(server_fd);
    glm_engine_close(&engine);
    return 0;
}