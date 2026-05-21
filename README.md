# ds4-glm: Custom Native GLM Engine

This branch is for a DS4-style custom native implementation of
GLM-4.7-Flash. The model backend does **not** use llama.cpp, vLLM, SGLang,
Ollama, or any existing inference engine.

LiteLLM is still used as the outer proxy for VS Code / GitHub Copilot plugin
compatibility:

```text
VS Code / Copilot plugin -> LiteLLM (11111) -> glm-native-server (11112)
```

Current status:

- Native C GGUF reader: implemented.
- Native GLM engine object: implemented.
- Native CLI loader: implemented.
- Native HTTP server shell: implemented.
- LiteLLM proxy to native backend: implemented.
- Tokenizer, forward pass, sampling, and CUDA kernels: not implemented yet.

The first real target model is:

```text
models/GLM-4.7-Flash-Q8_0.gguf
```

`Q8_0` is intentionally the default because it is the best foundation for a
custom engine port. It is larger than Q4/UD quants, but much simpler to
validate before implementing lower-bit kernels.

## Setup

```bash
cd /Users/sna/Desktop/projects/ds4-glm
./install.sh
```

`install.sh` runs the full local setup:

```bash
./setup-dev-venv.sh
./ensure-model.sh Q8_0
make -C native
```

`ensure-model.sh` first checks whether `GLM-4.7-Flash-Q8_0.gguf` already exists
in the Hugging Face cache on the DGX Spark. If present, it symlinks it into
`./models/`. If missing, it downloads it and then creates the same symlink.

Neither setup path installs an inference runtime. Native inference is built from
the C code under `native/`; LiteLLM is only the VS Code/Copilot proxy.

## Native Commands

Inspect GGUF metadata and tensor table:

```bash
./native-inspect.sh
./native/bin/glm-inspect --metadata --tensors models/GLM-4.7-Flash-Q8_0.gguf
```

Load the model through the native engine:

```bash
./start-gb10.sh inspect
```

Start native backend plus LiteLLM proxy:

```bash
./run-server-litellm.sh
```

Start only the native HTTP server shell:

```bash
./start-gb10.sh server
```

Health check:

```bash
curl http://localhost:11112/health
```

Model list:

```bash
curl http://localhost:11112/v1/models
```

Chat/completions endpoints currently return `501 not_implemented` until the
native forward pass exists.

LiteLLM endpoint for VS Code / Copilot-style clients:

```text
http://localhost:11111/v1/chat/completions
```

## Native Layout

| File | Purpose |
| --- | --- |
| `native/glm_gguf.c` | Minimal GGUF metadata/tensor reader. |
| `native/glm_engine.c` | Native model-load boundary and GLM metadata summary. |
| `native/glm_inspect.c` | GGUF inspector and validator. |
| `native/glm_native_cli.c` | DS4-style native CLI shell. |
| `native/glm_native_server.c` | DS4-style native HTTP server shell. |
| `server_compress.py` | LiteLLM proxy entrypoint for VS Code/Copilot compatibility. |
| `native/Makefile` | Builds all native binaries. |

Generated binaries:

```text
native/bin/glm-inspect
native/bin/glm-native
native/bin/glm-native-server
```

## Why LiteLLM But Not llama-server?

`llama-server` is an inference engine, so it is not used by the default path.
LiteLLM is not doing inference here; it is only the OpenAI-compatible proxy that
lets VS Code / GitHub Copilot plugin traffic reach the local native server.

The native GLM port needs its own:

- GLM tensor-name map.
- GLM tokenizer/chat-template path.
- CPU reference forward pass.
- MoE router and expert matmul.
- MLA attention and KV layout.
- CUDA kernels for GB10.
- Sampling and streaming server integration.

## Milestones

1. **GGUF load and validation**: current branch.
2. **Tensor map**: map all GLM tensors into typed layer structures.
3. **Tokenizer parity**: render/tokenize chat like the GGUF template expects.
4. **CPU first-token forward pass**: correctness before speed.
5. **Logit validation**: compare against a trusted runtime for fixed prompts.
6. **CUDA GB10 kernels**: attention, MoE, dense projections, norm, sampling.
7. **Full server generation**: OpenAI-compatible streaming completions.

## Model Source

The recommended model source is Unsloth's GGUF repo:

```text
unsloth/GLM-4.7-Flash-GGUF
```

The original `GadflyII/GLM-4.7-Flash-NVFP4` checkpoint is Safetensors /
compressed-tensors NVFP4 and is not the native target for this repo.
