# Custom Native GLM Engine — Quick Start

This repository builds a **custom native inference engine** for GLM-4.7-Flash from scratch,
without llama.cpp, vLLM, SGLang, or Ollama.

The stack is:

```
VS Code / Copilot → LiteLLM proxy (:11111) → native GLM server (:11112)
```

## What Exists vs. What's Stubbed

| Feature | Status |
|---|---|
| GGUF model loading (Q8_0) | ✅ Works |
| All 47 layer weight dequantization | ✅ Works |
| Native HTTP server (`glm-native-server`) | ✅ Works |
| `/health` and `/v1/models` endpoints | ✅ Works |
| **Tokenization** | ❌ Missing |
| **Forward pass (`glm_forward`)** | ⚠️ Partial (O(n²) CPU, not wired into server) |
| **Generation loop** | ⚠️ Written but not integrated into engine |
| **CUDA / GB10 kernels** | ❌ Not started |
| `/v1/chat/completions` | ❌ Returns 501 |

## Required Steps to Run

### 1. Install Python dependencies (venv + LiteLLM)

```bash
./setup-dev-venv.sh
```

Creates `venv/` with `huggingface_hub`, `litellm`, `fastapi`, `uvicorn`.

---

### 2. Download the Q8_0 GGUF model

```bash
./ensure-model.sh Q8_0
```

- First run: downloads from HuggingFace (`unsloth/GLM-4.7-Flash-GGUF`) into HF cache, then symlinks into `models/GLM-4.7-Flash-Q8_0.gguf`.
- Subsequent runs: detects cache hit and symlinks without re-downloading.

To use a different quantization:

```bash
./ensure-model.sh Q4_K_XL
./ensure-model.sh UD-Q4_K_XL
```

---

### 3. Build the native C binaries

```bash
make -C native
```

Or with custom compiler/flags:

```bash
CC=gcc CFLAGS="-O3 -march=native" make -C native
```

Produces:

```
native/bin/glm-inspect       # GGUF metadata/tensor inspector
native/bin/glm-native        # CLI loader with model summary
native/bin/glm-native-server # HTTP server (generation = 501)
```

---

### 4. Inspect the model (no generation)

```bash
# Summary + GLM validation
./start.sh inspect

# Full metadata
./native/bin/glm-inspect --metadata models/GLM-4.7-Flash-Q8_0.gguf

# Tensor listing
./native/bin/glm-inspect --tensors models/GLM-4.7-Flash-Q8_0.gguf
```

The inspector validates layer counts, context length, MoE metadata, and tensor naming conventions.

---

### 5. Start the native HTTP server (generation returns 501)

```bash
./start.sh server
```

Or via the combined launcher:

```bash
./run-server-litellm.sh server
```

Server binds to `0.0.0.0:11112` by default. Endpoints:

```
GET  http://localhost:11112/health       # {"status":"ok","generation":false}
GET  http://localhost:11112/v1/models   # model card JSON
POST http://localhost:11112/v1/chat/completions  # 501 Not Implemented
```

---

### 6. Start LiteLLM proxy (for VS Code / Copilot)

```bash
./run-server-litellm.sh proxy
```

Binds to `0.0.0.0:11111`. Forwards to the native server at `localhost:11112`.

```bash
# Health check
curl http://localhost:11111/health

# OpenAI-compatible endpoint (will also return 501)
curl http://localhost:11111/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"Hi"}]}'
```

---

### 7. One-command setup + server startup

```bash
# Full install (venv + model + build + start)
./install.sh
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GLM_MODEL` | `models/GLM-4.7-Flash-Q8_0.gguf` | Path to GGUF file |
| `GLM_HOST` | `0.0.0.0` | Native server bind host |
| `GLM_PORT` | `11112` | Native server port |
| `LITE_LLM_PROXY_HOST` | `0.0.0.0` | LiteLLM proxy bind host |
| `LITE_LLM_PROXY_PORT` | `11111` | LiteLLM proxy port |
| `GLM_GGUF_QUANT` | `Q8_0` | Quantization variant (for `ensure-model.sh`) |
| `HF_TOKEN` | *(empty)* | HuggingFace token (optional) |

---

## All-In-One Script Reference

| Script | What it does |
|---|---|
| `./setup-dev-venv.sh` | Create/update `venv/` with Python deps |
| `./ensure-model.sh [QUANT]` | Download/symlink GGUF into `models/` |
| `./download_model.sh [QUANT]` | Alias for `ensure-model.sh` |
| `./install.sh [QUANT]` | Full setup: venv + model + build |
| `./native-inspect.sh` | Build + run `glm-inspect` |
| `./start.sh [mode]` | Build + run native server (modes: `both`, `server`, `proxy`, `inspect`, `summary`) |
| `./run-server-litellm.sh [mode]` | Combined native + LiteLLM launcher |
| `./kill.sh` | Stop all running server processes |

---

## Architecture Overview

```
native/glm_gguf.c       — GGUF metadata + tensor table reader (Q8_0 dequant on load)
native/glm_tensor.c     — Weight tensor structures + 3D MoE/attn layout remapping
native/glm_fwd.c        — Forward pass: MLA attention, MoE FFN, RMSNorm, RoPE, sampling
native/glm_engine.c     — Model-open/close boundary, metadata summary
native/glm_inspect.c    — CLI: --metadata, --tensors, GLM validation
native/glm_native_cli.c — CLI: load model, print summary, generation (stubbed)
native/glm_native_server.c — HTTP server: /health, /v1/models, /v1/chat/completions (501)
native/Makefile         — Builds all binaries
server_compress.py      — LiteLLM proxy entrypoint
```

---

## What Needs to Be Done to Get Generation Working

1. **Tokenization**: Wire a tokenizer (SentencePiece or HuggingFace tokenizer) into `glm_engine` — the server needs to convert text → token IDs before calling `glm_forward`.
2. **Forward pass integration**: `glm_engine_generate_text()` in `glm_engine.c` is a stub that returns "not implemented". It needs to call `glm_forward` from `glm_fwd.c`.
3. **KV cache**: The current `glm_fwd.c` does a full O(n²) attention pass on every call. For decoding, it needs to use the `kv_cache` structure to cache K/V across tokens.
4. **Server integration**: `glm_native_server.c` returns 501 on `/v1/chat/completions` — needs to call `glm_engine_generate_text/stream` and format the response.
5. **Optional**: Replace pure-C GEMM with BLAS (OpenBLAS/MKL) or SIMD kernels for meaningful throughput.
6. **Optional**: CUDA kernels for GB10 NVFP4 inference.

The forward pass and generation loop **are written** in `glm_fwd.c` — they just need to be connected to the server and validated against a trusted runtime.