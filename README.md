# spermC v2

`spermC` is a single-file GGUF inference engine aimed at old Windows targets, including Windows 2000, while still being practical to build and test on modern Windows hosts.

## What it is

- Native local GGUF inference
- Windows 2000-friendly C codebase
- Single executable: `spermC.exe`
- Model-side prompt handling for several common instruct/chat formats
- Optional OpenAI-compatible API mode

## Current focus

This project is optimized around:

- old Windows compatibility first
- CPU inference only
- small and medium GGUF models
- simple local usage without extra runtimes

It is not a full `llama.cpp` replacement. It is a compact native runner with targeted support for the models used in this repo.

## Supported model families

The runtime has native paths for these architectures:

- Llama-family models
- TinyLlama
- Llama 3 / 3.1 / 3.2 Instruct
- SmolLM2
- Qwen2 / Qwen2.5 / Qwen3
- Gemma / Gemma 2 / Gemma 3
- GPT-2

GGUF tensor and tokenizer handling is auto-detected where possible.

## Quantization support

Supported tensor formats include:

- `F32`
- `F16`
- `Q4_0`
- `Q5_0`
- `Q5_1`
- `Q8_0`
- `Q4_K`
- `Q5_K`
- `Q6_K`

## Features

- Loads tokenizer data from GGUF metadata when present
- Falls back to external tokenizer discovery with `--tok auto`
- Sorted tokenizer lookup for faster token matching
- Multi-threaded CPU matvec on WinNT-class systems
- F32 caching of hot tensors when memory budget allows
- Interactive mode that keeps the model loaded between turns
- Timing stats including TTFT and TPS
- Optional clean output mode with `--clean`

## Build

Run:

```bat
build.bat
```

The build script tries:

- `cl.exe` first
- `gcc.exe` second

Output binary:

```bat
spermC.exe
```

## Usage

```bat
spermC.exe <model.gguf> [options]
```

## CLI options

- `--info` Dump model info and exit
- `--list` List tensors and exit
- `--clean` Suppress non-essential output
- `--no-eos-stop` Do not stop generation on EOS
- `--raw-tokens` Print token IDs instead of decoded text
- `--chat` Wrap the prompt in a chat template
- `--repl` Keep the model loaded and read prompts from stdin
- `--keep-loaded` Alias for `--repl`
- `--interactive` Llama.cpp-style interactive chat; stops on EOS
- `--api` Use an OpenAI-compatible API instead of local inference
- `--api-url <url>` Override the API endpoint
- `--api-model <name>` Override the API model name
- `--api-key <key>` Override the API key
- `--api-system <text>` System prompt for API mode
- `--system <text>` System prompt for local chat templates
- `-f <file>` Read prompt text from a file
- `--prompt <text>` Prompt text inline
- `-n <num>` Max tokens to generate
- `-t <temp>` Temperature
- `-s <seed>` RNG seed
- `--tok <file|auto>` External tokenizer path or auto-discovery
- `--n_heads <n>` Override attention head count
- `--n_kv <n>` Override KV head count
- `--hidden <n>` Override FFN hidden size
- `--seq <n>` Override max sequence length
- `--top-k <n>` Top-k sampling
- `--top-p <p>` Top-p sampling
- `--min-p <p>` Min-p sampling
- `--eos <id>` Override EOS token ID
- `--repeat-penalty <p>` Repetition penalty
- `--threads <n|auto>` CPU thread count for large matvec

## Modes

### One-shot local inference

```bat
spermC.exe C:\path\to\model.gguf --prompt "hi" -n 16 --threads auto
```

### Interactive mode

This keeps the model loaded, which avoids paying full load cost every turn.

```bat
spermC.exe C:\path\to\model.gguf --interactive --threads auto
```

### REPL mode

This also keeps the model loaded, but reads prompts from stdin without the interactive chat framing behavior.

```bat
spermC.exe C:\path\to\model.gguf --repl --threads auto
```

### API mode

```bat
set OPENAI_API_KEY=your_key_here
spermC.exe dummy.gguf --api --api-model gpt-4o-mini --prompt "hi"
```

`dummy.gguf` still occupies the model-path slot in the CLI. In API mode, the request is sent to the configured endpoint instead of doing local GGUF inference.

## Model-specific behavior

### TinyLlama

- TinyLlama-style chat wrapping is auto-detected from the model name
- Useful for small instruct-style TinyLlama GGUFs

Example:

```bat
spermC.exe C:\Users\maxwe\Downloads\TinyLlama-v0.Q8_0.gguf --prompt "hi" -n 8 -t 0 --threads auto
```

### SmolLM2

- SmolLM2 instruct models work best with chat-style prompting
- Very small models can still produce weak answers even when the runtime path is correct

Example:

```bat
spermC.exe C:\Users\maxwe\Downloads\SmolLM2-360M-Instruct.gguf --prompt "what's 2+2?" -n 32 --threads auto
```

### Qwen

- Qwen-family chat formatting is handled natively
- Embedded tokenizer support matters here; mismatched tokenization will break outputs

Example:

```bat
spermC.exe C:\Users\maxwe\Downloads\Qwen3-0.6B-Q4_K_M-Instruct.gguf --prompt "hi" -n 16 --threads auto
```

### Llama 3.2 Instruct

- Llama 3.x instruct-style header templates are detected from GGUF metadata
- Local wrapping uses:
- `"<|begin_of_text|><|start_header_id|>system<|end_header_id|> ... <|eot_id|><|start_header_id|>user<|end_header_id|> ... <|eot_id|><|start_header_id|>assistant<|end_header_id|>"`
- `--system` fills the `{system_prompt}` slot for local inference
- `--repl` and `--interactive` cache the fixed Llama 3 prefix so later prompts do much less prefill work

Example:

```bat
spermC.exe C:\Users\maxwe\Downloads\Llama-3.2-1B-Instruct-Q4_K_M.gguf --system "You are concise." --prompt "hi" -n 16 --threads auto
```

### Gemma 3

- Gemma 3 uses start-of-turn / end-of-turn chat formatting
- Default sampler settings are applied unless you override them:
- `temperature = 1.0`
- `top_k = 64`
- `top_p = 0.95`
- `min_p = 0.0`

Example:

```bat
spermC.exe C:\Users\maxwe\Downloads\gemma-3-270m-it-Q6_K.gguf --prompt "hi" -n 12 --threads auto --temp 0
```

### GPT-2

- GPT-2 support is included for compatible GGUF exports
- Expect completion-style behavior, not modern instruct behavior, unless the model itself was instruction-tuned

Example:

```bat
spermC.exe C:\Users\maxwe\Downloads\gpt2-instruct.gguf --prompt "hi" -n 32 --threads 4
```

## Windows 2000 notes

Windows 2000 is the hardest target here.

- Old systems are much more memory-sensitive than modern hosts
- Large F32 caches that help speed on Windows 11 can cause paging on low-RAM Win2000 VMs
- The runtime now uses a conservative legacy cache budget based on physical RAM
- On a `1 GB` Win2000 machine, expect smaller caches and lower TPS than on a modern host

If Windows starts growing the page file or the system becomes unresponsive:

- use a smaller model
- reduce `--threads`
- reduce `--seq`
- avoid giant Gemma/Qwen models on very low RAM

## Performance notes

- `--threads auto` is usually the right starting point on modern hosts
- On old systems, explicit thread counts like `--threads 2` or `--threads 4` may behave better
- TTFT depends heavily on prompt length, cache state, model size, and whether the model stays loaded
- Interactive or REPL mode is much faster across multiple turns than launching the executable for every question

## Output quality notes

If output looks like garbage, the most common causes are:

- wrong prompt format for the model
- tokenizer mismatch
- bad GGUF export
- tiny model limitations
- too-aggressive threading or old-system instability

For instruct models, prefer:

- `--interactive`
- `--chat` when appropriate
- a model family the runtime already handles natively

## Examples

### Fast local single prompt

```bat
spermC.exe C:\Users\maxwe\Downloads\smollm2-360m-instruct-q8_0.gguf --prompt "can you say OK sure?" -n 30 --threads auto
```

### Interactive Gemma 3

```bat
spermC.exe C:\Users\maxwe\Downloads\gemma-3-270m-it-Q6_K.gguf --interactive --threads auto --temp 0
```

### Low-RAM Win2000-style run

```bat
spermC.exe gemma-3-270m-it-Q6_K.gguf --prompt "hi" -n 8 --threads 4 --temp 0
```

## Project status

This is an actively hacked local runtime, not a polished general-purpose framework.

Expect:

- targeted model fixes
- pragmatic compatibility work
- model-specific prompt handling
- performance tuning that depends on the exact machine and GGUF
