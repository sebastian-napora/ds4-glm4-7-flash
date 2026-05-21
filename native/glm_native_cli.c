#include "glm_engine.h"
#include "glm_native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s -m MODEL.gguf [OPTIONS]\n"
            "\n"
            "Options:\n"
            "  -p, --prompt TEXT   Try one generation request (currently not implemented)\n"
            "  --summary           Print model summary after load (default)\n"
            "  -h, --help          Show this help\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *prompt = NULL;
    bool summary = true;

    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) {
            model = argv[++i];
        } else if ((!strcmp(argv[i], "-p") || !strcmp(argv[i], "--prompt")) && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "--summary")) {
            summary = true;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!model) {
        usage(argv[0]);
        return 2;
    }

    printf("glm-native %s\n", GLM_NATIVE_VERSION);

    glm_engine engine;
    if (!glm_engine_open(&engine, model)) {
        fprintf(stderr, "glm-native: %s\n", engine.error[0] ? engine.error : "failed to open model");
        glm_engine_close(&engine);
        return 1;
    }

    if (summary) glm_engine_print_summary(&engine, stdout);

    if (prompt) {
        char out[512];
        int rc = glm_engine_generate_text(&engine, prompt, out, sizeof(out));
        fprintf(stderr, "%s\n", out);
        glm_engine_close(&engine);
        return rc == 0 ? 0 : 4;
    }

    glm_engine_close(&engine);
    return 0;
}
