# ds4-glm: GLM-4.7-Flash-NVFP4 on DGX Spark GB10

This folder is a GLM-specific sibling setup inspired by `ds4-changed`, but it
does not reuse the native DS4 GGUF engine. `GadflyII/GLM-4.7-Flash-NVFP4` is a
Safetensors `glm4_moe_lite` checkpoint using compressed-tensors NVFP4, so the
runtime here is:

```text
client / agent -> LiteLLM proxy (11111) -> vLLM OpenAI server (11112)
                                      \-> token stats server (11113)
```

The intended machine is a DGX Spark / GB10 class box with 128 GB unified memory.

## Why This Is Different From ds4-changed

`ds4-changed` is a narrow custom runner for DeepSeek V4 Flash GGUF files. Its
CUDA launcher can lazily resident-cache routed expert slices with
`DS4_CUDA_LAZY_ROUTED_EXPERTS`.

GLM-4.7-Flash-NVFP4 is different:

- Model format: Safetensors + compressed-tensors NVFP4, not GGUF.
- Architecture: `Glm4MoeLiteForCausalLM`, 30B total / 3B active.
- MoE shape: 64 routed experts, 4 active per token, 1 shared expert.
- Attention: MLA is left BF16 in this quant; the FP4 path mostly covers MLPs.
- Runtime: vLLM owns the compressed-tensors/NVFP4 kernels and OpenAI API.

So the equivalent of the DS4 "kernel approach" is not a C engine port. It is a
careful vLLM launch profile:

- `VLLM_USE_FLASHINFER_MOE_FP4=0` on GB10, because FlashInfer FP4 MoE has been
  unstable on this hardware path in the copied setup.
- `VLLM_OPT_LEVEL=1` by default: CUDA graphs without torch.compile.
- `VLLM_GPU_MEM_UTIL=0.50` by default for 128 GB headroom.
- `VLLM_MAX_MODEL_LEN=65536` by default for agent work; raise to `202752` only
  when you need full context.
- `--tool-call-parser glm47` and `--reasoning-parser glm45` for GLM chat/tool
  semantics.

## Files

| File | Purpose |
| --- | --- |
| `setup-venv.sh` | Creates `venv` and installs serving requirements. |
| `download_model.sh` | Downloads the HF snapshot into `./model`. |
| `start-gb10.sh` | Recommended GB10 launcher with conservative defaults. |
| `start-low-gpu.sh` | Lower memory/power profile for shared use. |
| `start.sh` | Starts backend, proxy, and token stats services. |
| `glm_server.py` | vLLM OpenAI-compatible backend. |
| `server_compress.py` | LiteLLM proxy with GLM history sanitization callbacks. |
| `glm_compress.py` | Strips/thins reasoning, tool schemas, and long tool results. |
| `glm_token_tracker.py` | SQLite-backed token usage callback. |
| `token_stats_server.py` | Small stats UI/API on port 11113. |
| `start_ngram.sh` | Enables vLLM prompt-lookup/n-gram speculative decoding. |
| `start_mtp.sh` | Guarded off; this NVFP4 checkpoint currently breaks vLLM MTP. |
| `kill.sh` | Stops local GLM/vLLM/LiteLLM processes and frees ports. |

## Quick Start

```bash
cd /Users/sna/Desktop/projects/ds4-glm
./setup-venv.sh
./download_model.sh
./start-gb10.sh both
```

If you prefer not to download a local snapshot first, `start-gb10.sh` will load
directly from `GadflyII/GLM-4.7-Flash-NVFP4` through the Hugging Face cache.

Health checks:

```bash
curl http://localhost:11112/health
curl http://localhost:11111/health
```

OpenAI-compatible chat request:

```bash
curl http://localhost:11111/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7-flash-nvfp4",
    "messages": [{"role": "user", "content": "Explain KV cache in two sentences."}],
    "max_tokens": 200
  }'
```

## GB10 Profiles

Recommended default:

```bash
./start-gb10.sh both
```

Conservative shared-machine profile:

```bash
./start-low-gpu.sh both
```

Full context attempt:

```bash
VLLM_MAX_MODEL_LEN=202752 VLLM_GPU_MEM_UTIL=0.70 ./start-gb10.sh both
```

Minimal RAM/cold-start profile:

```bash
VLLM_OPT_LEVEL=0 VLLM_MAX_MODEL_LEN=32768 VLLM_GPU_MEM_UTIL=0.45 ./start-gb10.sh both
```

N-gram speculative decoding for long prompts/RAG/code tasks:

```bash
./start_ngram.sh both
```

## Environment Knobs

| Variable | Default | Notes |
| --- | --- | --- |
| `GLM_MODEL_PATH` | HF repo or `./model` if present | Local snapshot or HF repo id. |
| `GLM_SERVED_MODEL_NAME` | `GadflyII/GLM-4.7-Flash-NVFP4` | Name accepted by vLLM backend. |
| `VLLM_MAX_MODEL_LEN` | `65536` via `start-gb10.sh` | Native max is `202752`. |
| `VLLM_GPU_MEM_UTIL` | `0.50` | Increase for full context. |
| `VLLM_OPT_LEVEL` | `1` | `0` eager, `1` CUDA graphs, avoid `3` on this model. |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `65536` | Reduce if startup memory is tight. |
| `GLM_PORT` | `11112` | vLLM backend. |
| `LITE_LLM_PROXY_PORT` | `11111` | LiteLLM proxy. |
| `GLM_TOKEN_STATS_PORT` | `11113` | Stats server. |

## MTP Status

The model config has `num_nextn_predict_layers: 1`, but the current
`GadflyII/GLM-4.7-Flash-NVFP4` checkpoint NVFP4-quantizes the MTP `eh_proj`
linear while vLLM's GLM MTP implementation expects it unquantized. The guarded
`start_mtp.sh` script exits with an explanation instead of spending minutes
loading and then failing with:

```text
KeyError: 'model.layers.47.eh_proj.input_global_scale'
```

Use the normal stack or `start_ngram.sh` until vLLM/checkpoint support changes.

## Model Facts

- Hugging Face repo: <https://huggingface.co/GadflyII/GLM-4.7-Flash-NVFP4>
- Base model: `zai-org/GLM-4.7-Flash`
- Architecture: `Glm4MoeLiteForCausalLM`
- Parameters: 30B total, 3B active per token
- Experts: 64 routed, 4 active, 1 shared
- Context: 202,752 tokens
- Quantization: compressed-tensors NVFP4, mixed precision
- License: Apache 2.0
