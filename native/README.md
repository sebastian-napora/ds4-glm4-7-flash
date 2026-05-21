# GLM Native Skeleton

This directory is the first milestone for a DS4-style GLM implementation.

It deliberately stops before inference. The engine loads the GGUF directly,
exposes a CLI and HTTP server shell, and keeps generation disabled until the
GLM forward pass is implemented.

## Build

```bash
cd native
make
```

## Inspect A GGUF

```bash
./bin/glm-inspect ../models/GLM-4.7-Flash-Q8_0.gguf
./bin/glm-inspect --metadata --tensors ../models/GLM-4.7-Flash-Q8_0.gguf
./bin/glm-native -m ../models/GLM-4.7-Flash-Q8_0.gguf
./bin/glm-native-server -m ../models/GLM-4.7-Flash-Q8_0.gguf --host 0.0.0.0 --port 11112
```

Exit codes:

- `0`: parsed and basic GLM validation passed
- `1`: GGUF parse/read failure
- `2`: CLI usage error
- `3`: parsed, but validation emitted warnings

## What This Proves

- The file is valid GGUF.
- The metadata table can be read.
- The tensor table can be read.
- The model looks GLM-like enough to begin engine work.

## Next Milestones

1. Build a GLM tensor map from GGUF tensor names.
2. Add tokenizer/chat template tests against llama.cpp output.
3. Implement CPU reference first-token forward pass.
4. Compare logits against llama.cpp or Transformers.
5. Port hot paths to CUDA for GB10.
