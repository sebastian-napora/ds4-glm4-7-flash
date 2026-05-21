CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS ?=
LDFLAGS ?=

BIN_DIR := bin
INSPECT := $(BIN_DIR)/glm-inspect
CLI := $(BIN_DIR)/glm-native
SERVER := $(BIN_DIR)/glm-native-server
COMMON_SRCS := glm_gguf.c glm_engine.c
COMMON_OBJS := $(COMMON_SRCS:.c=.o)

.PHONY: all clean

all: $(INSPECT) $(CLI) $(SERVER)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(INSPECT): glm_inspect.o $(COMMON_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ glm_inspect.o $(COMMON_OBJS)

$(CLI): glm_native_cli.o $(COMMON_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ glm_native_cli.o $(COMMON_OBJS)

$(SERVER): glm_native_server.o $(COMMON_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ glm_native_server.o $(COMMON_OBJS)

%.o: %.c glm_gguf.h glm_native.h glm_engine.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -rf $(BIN_DIR)
