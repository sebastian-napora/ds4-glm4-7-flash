#include "glm_engine.h"
#include "glm_native.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define REQ_MAX 65536

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s -m MODEL.gguf [--host HOST] [--port PORT]\n"
            "\n"
            "Endpoints:\n"
            "  GET  /health\n"
            "  GET  /v1/models\n"
            "  POST /v1/chat/completions  (501 until generation is implemented)\n",
            argv0);
}

static bool send_all(int fd, const char *buf, size_t len) {
    while (len) {
        ssize_t n = send(fd, buf, len, 0);
        if (n <= 0) return false;
        buf += n;
        len -= (size_t)n;
    }
    return true;
}

static void http_json(int fd, int status, const char *reason, const char *body) {
    char header[512];
    const int body_len = (int)strlen(body);
    int n = snprintf(header, sizeof(header),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: application/json\r\n"
                     "Content-Length: %d\r\n"
                     "Connection: close\r\n"
                     "\r\n",
                     status, reason, body_len);
    send_all(fd, header, (size_t)n);
    send_all(fd, body, (size_t)body_len);
}

static void handle_client(int fd, glm_engine *engine) {
    char req[REQ_MAX + 1];
    ssize_t n = recv(fd, req, REQ_MAX, 0);
    if (n <= 0) return;
    req[n] = 0;

    char method[16] = {0};
    char path[256] = {0};
    sscanf(req, "%15s %255s", method, path);

    if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        char body[1024];
        snprintf(body, sizeof(body),
                 "{\"status\":\"ok\",\"engine\":\"glm-native\",\"version\":\"%s\","
                 "\"model\":\"%s\",\"generation\":false}\n",
                 GLM_NATIVE_VERSION, glm_engine_model_id(engine));
        http_json(fd, 200, "OK", body);
        return;
    }

    if (!strcmp(method, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/v1/models/"))) {
        char body[2048];
        snprintf(body, sizeof(body),
                 "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
                 "\"owned_by\":\"local\",\"root\":\"%s\"}]}\n",
                 glm_engine_model_id(engine), glm_engine_model_id(engine));
        http_json(fd, 200, "OK", body);
        return;
    }

    if (!strcmp(method, "POST") &&
        (!strcmp(path, "/v1/chat/completions") || !strcmp(path, "/v1/completions"))) {
        char body[1024];
        snprintf(body, sizeof(body),
                 "{\"error\":{\"message\":\"GLM native generation is not implemented yet; "
                 "the custom engine currently loads the GGUF model and exposes the native "
                 "server shell only.\",\"type\":\"not_implemented\",\"code\":\"glm_native_forward_missing\"}}\n");
        http_json(fd, 501, "Not Implemented", body);
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
            port = atoi(argv[++i]);
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
    if (server_fd < 0) {
        perror("socket");
        glm_engine_close(&engine);
        return 1;
    }

    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (!strcmp(host, "0.0.0.0")) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "invalid IPv4 host: %s\n", host);
        close(server_fd);
        glm_engine_close(&engine);
        return 1;
    }

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        close(server_fd);
        glm_engine_close(&engine);
        return 1;
    }
    if (listen(server_fd, 16) != 0) {
        perror("listen");
        close(server_fd);
        glm_engine_close(&engine);
        return 1;
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
