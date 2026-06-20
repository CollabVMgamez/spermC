# windows2000ai

Single-file GGUF inference engine for old Windows targets, built and tested on Windows 11 host machines.

## What it does

- Loads local `.gguf` models with embedded tokenizer metadata when available
- Supports `F32`, `F16`, `Q4_0`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K`
- Handles Llama-style model metadata and tensor naming
- Auto-detects transposed token embeddings
- Caches hot weights in `F32` for speed on small and medium models
- Uses multi-threaded matmul for large row-major tensors
- Auto-detects TinyLlama and uses a Llama-2 style instruction prompt wrapper

## Build

Run:

```bat
build.bat
```

The script uses `cl.exe` if available, otherwise `gcc.exe`.

## Usage

```bat
gguf_infer.exe <model.gguf> [options]
```

Common options:

- `--info` Dump model info and exit
- `--list` List all tensors and exit
- `--chat` Use a chat-style prompt wrapper
- `--prompt <text>` Prompt string
- `-f <file>` Read prompt from file
- `-n <num>` Max tokens to generate
- `-t <temp>` Temperature, `0` = argmax
- `-s <seed>` Random seed
- `--tok <file|auto>` Tokenizer file or auto-discover
- `--threads <n|auto>` Parallel threads for large matmul
- `--raw-tokens` Print token IDs instead of text
- `--clean` Reduce logging noise

## Example

TinyLlama:

```bat
gguf_infer.exe C:\Users\maxwe\Downloads\TinyLLama-v0.Q8_0.gguf --prompt hi -n 8 -t 0 --threads auto
```

SmolLM2:

```bat
gguf_infer.exe C:\Users\maxwe\Downloads\SmolLM2-135M-Instruct-Q4_K_M.gguf --prompt hi -n 8 -t 0 --threads auto
```

## Notes

- TinyLlama files are auto-wrapped in a Llama-2 style instruction prompt when the model path contains `tinyllama`.
- If a model has transposed token embeddings or output weights, the runtime caches them in `F32` to avoid repeated dequantization.
- Very small prompts can still look odd if the model itself is low quality or the prompt format is wrong. Use `--chat` for chat/instruct models when needed.

