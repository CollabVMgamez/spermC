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
- Auto-detects Gemma 3 style prompts and applies `temperature=1.0`, `top_k=64`, `top_p=0.95`, `min_p=0.0` unless overridden
- Can switch to an OpenAI-compatible API backend with `--api`

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
- `--api` Use an OpenAI-compatible API instead of local inference
- `--api-url <url>` API endpoint URL
- `--api-model <name>` API model name
- `--api-key <key>` API key
- `--api-system <text>` System prompt for API mode
- `--prompt <text>` Prompt string
- `-f <file>` Read prompt from file
- `-n <num>` Max tokens to generate
- `-t <temp>` Temperature, `0` = argmax
- `--top-k <n>` Top-k sampling
- `--top-p <p>` Top-p sampling
- `--min-p <p>` Min-p sampling
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

Gemma 3:

```bat
gguf_infer.exe C:\Users\maxwe\Downloads\gemma-3-270m-it-Q4_K_M.gguf --prompt hi -n 8 --threads auto
```

API mode:

```bat
set OPENAI_API_KEY=...
gguf_infer.exe gemma-3-270m-it-Q4_K_M --api --api-model gpt-4o-mini --prompt hi -n 8
```

## Notes

- TinyLlama files are auto-wrapped in a Llama-2 style instruction prompt when the model path contains `tinyllama`.
- Gemma 3 files are wrapped with `<start_of_turn>` / `<end_of_turn>` formatting.
- If a model has transposed token embeddings or output weights, the runtime caches them in `F32` to avoid repeated dequantization.
- Very small prompts can still look odd if the model itself is low quality or the prompt format is wrong. Use `--chat` for chat/instruct models when needed.
