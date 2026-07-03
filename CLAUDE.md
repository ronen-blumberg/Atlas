# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Atlas is a character-level GPT (Transformer) language model implemented **from scratch in pure FreeBASIC** — no external ML libraries. Everything (tokenizer, forward pass, hand-derived backward pass, AdamW, multi-threaded training loop, chat) lives in the single file `atlas.bas` (~1000 lines). It builds from the same source on Linux 64-bit and Windows 32-bit.

Published at https://github.com/ronen-blumberg/Atlas.

## Build & run

```bash
./build_linux.sh              # Linux 64-bit
# expands to: fbc -gen gcc -O 3 -mt -Wc -march=native,-funroll-loops atlas.bas -x atlas
```
Windows 32-bit: `build_windows.bat` (same flags minus `-march=native`).

Three flags are load-bearing and must not be dropped:
- **`-mt`** — multi-threaded runtime. Training uses `ThreadCreate`/`ThreadWait`; without `-mt` the runtime is not thread-safe and it will crash.
- **`-gen gcc -O 3`** — the gcc backend is several times faster than FreeBASIC's default backend for the scalar-float inner loops. The default backend is unusably slow for training.

```bash
./atlas train [steps]   # train from data/corpus.txt, checkpoint model.bin every 500 steps
./atlas chat            # load model.bin, interactive chat (also works with piped stdin)
./atlas                 # chat if model.bin exists, else train
```

Regenerate the training corpus (seeded/deterministic, stdlib-only Python):
```bash
python3 scripts/gen_corpus.py 1000000 data/corpus.txt
```
`data/corpus.txt` (~60 MB) is **gitignored** — a fresh clone has no corpus, so run this before any training. The local `wordnet/` directory is unused by the corpus generator (leftover data, also gitignored); don't look there for the persona sources.

There is no test suite. Verify changes empirically: a short `./atlas train 300` should show cross-entropy loss falling from ~ln(vocab) (~3.5–4) toward ~1; if it prints `loss 0.0000` or doesn't drop, a gradient or a loop is broken.

## Architecture (the parts that span files / aren't obvious)

Model: token+positional embeddings → `NLAYER` × (pre-LayerNorm → causal multi-head self-attention → residual, then pre-LayerNorm → MLP(Linear→GELU→Linear) → residual) → final LayerNorm → linear head → softmax. All hyperparameters are **compile-time `Const`s** at the top of `atlas.bas` (`BLOCK`, `NEMB`, `NHEAD`, `NLAYER`, `FFN`, `NTHREAD`, `BATCH`, learning-rate schedule). To resize the model you edit these and rebuild — the buffer dimensions are derived from them.

Key design decisions to understand before editing:

- **Parameter registry (`ParamT` / `params()` / `newParam`).** Every weight tensor is registered once with its data, a reduced gradient, per-thread gradients, and Adam moments. Initialization, the optimizer, save/load, and gradient reduction all iterate this list uniformly. Add a weight → register it in `buildModel` and it's automatically handled everywhere.

- **`Workspace` type = all per-sequence activations + backprop scratch.** There is one heap-allocated `Workspace` per worker thread (`ws()`), so threads never share activation memory. `forward`/`backward`/`logitsAt` all take `ByRef w As Workspace`.

- **Hand-derived backprop.** `backward` runs `forward`, computes softmax cross-entropy, then walks the network in reverse. Each op's gradient (LayerNorm, attention softmax, GELU, linear, embedding) is derived inline in comments. Gradients accumulate into `params(i).gradT(tid)` (this thread's buffer), never a shared buffer.

- **Data-parallel training.** `train` picks `BATCH` random windows in the main thread (keeps RNG deterministic), spawns `NTHREAD` `workerProc` threads that each process a stride of the batch into their own workspace + `gradT(tid)`, joins, then **reduces** all `gradT` into `params.grad` (averaged over batch) before one `adamStep`. Threads only read shared weights, so there are no data races and no locks.

- **Byte tokenizer.** `dataArr` is `UByte` (not `Integer`) so a large corpus (tens of MB) stays within 32-bit address limits. Vocabulary is built from whatever bytes appear in `data/corpus.txt` (≤256).

- **Checkpoint format** (`saveModel`/`loadModel`): magic `ATL4`, then `{BLOCK,NEMB,NHEAD,NLAYER,FFN,vocab}`, the vocab bytes, then raw param data. Loading aborts if the compiled dimensions don't match the file — so changing a `Const` invalidates old `model.bin`. The header ints are written as **`Long` (fixed 32-bit)**, not `Integer`, so the file is byte-identical across the Linux-64 and Win-32 builds and a `model.bin` is portable both directions (weights are `Single` = 4 bytes everywhere). Assumes little-endian (x86/x64 both are). Bump the magic (`ATL4` → `ATL5` …) whenever you change the on-disk layout, so old files fail the magic check cleanly instead of loading garbage.

- **Generation/chat** is single-threaded (uses `ws(0)`). Chat keeps a rolling `ctx()` and feeds the last `BLOCK` tokens per token generated; the corpus/chat protocol is the literal `User: …\nBot: …\n` line format.

## FreeBASIC gotchas that will bite (learned the hard way here)

- **`Integer` is native word width** — 8 bytes on 64-bit, 4 bytes on 32-bit. Anything written to disk (or shared across the Linux-64 / Win-32 builds) with `Integer` will misalign cross-platform. Use `Long` (always 32-bit) or another fixed-width type for on-disk headers; this is why the checkpoint header uses `Long`. `Single`/`Double` are fixed-width and safe.
- **Identifiers are case-INSENSITIVE.** A loop `For t As Integer = 0 To T-1` where `T` is also a parameter is the *same variable* `t==T`; the loop silently runs zero times (limit becomes `0-1`). Never let a loop variable and another identifier differ only in case. Reserved words that have caused errors: `step`, `line`, `base`.
- **Reading stdin is genuinely tricky in FB.** At startup FB's runtime puts the terminal in **raw mode (echo + canonical OFF)** so its own input functions can echo/line-edit manually. Consequences:
  - **`Line Input`/`Input`** echo correctly *interactively* (FB does the echo), but **hang on piped input** (they do an `ESC[6n` cursor-query handshake that a pipe never answers).
  - **`Open Cons For Input`** handles pipes but **breaks interactive typing** (no echo — user "can't type").
  - **`fgets`** handles pipes but is **silent interactively** (it reads the raw, echo-off terminal, so typed characters don't show).

  So `chat` **branches on `isatty(0)`**: FB `Line Input` for a real terminal, `fgets` (from `crt.bi`) for piped/redirected input. `isatty` isn't in `crt.bi` — it's declared manually via `Extern "C"`. Verify the interactive path by driving the binary through a pseudo-terminal (Python `pty`), answering the `ESC[6n` query; a plain pipe cannot exercise it.
- The shell working directory resets between tool calls in this environment; use absolute paths or `cd` inside a single command.

## Persona / data notes

The chatbot's character (warm older man: life wisdom, poetry, FreeBASIC Q&A, mental-health support) comes entirely from `data/corpus.txt`, generated by `scripts/gen_corpus.py` from themed fragment pools. The support content is **crisis-aware** — inputs signalling serious distress are answered by urging the user toward a trusted person / professional / crisis line. It is a companion voice, explicitly not a mental-health service; preserve that framing when editing the corpus generator.
