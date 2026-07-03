# Atlas — a small LLM trained from scratch in FreeBASIC

Atlas is a **character-level GPT** (a Transformer language model) written entirely
in FreeBASIC, with **no external ML libraries**. The tokenizer, the forward pass,
the full backward pass (every gradient derived by hand), the AdamW optimizer, the
**multi-threaded** training loop, and the interactive chat loop are all in one
source file: [`atlas.bas`](atlas.bas).

It builds from the same source on **Linux 64-bit** and **Windows 32-bit**.

Atlas has a **persona**: a warm older man with life experience — reflective and
kind, fond of poetry, supportive of mental well-being, and knowledgeable in
FreeBASIC. That character comes entirely from its training data
(`data/corpus.txt`), which you can replace.

> **What it is / isn't.** This is a genuine neural network that learns by gradient
> descent — but it's small (~2.7M parameters, 128 characters of memory) and
> character-level. It convincingly learns the *voice, tone, and short phrases* of
> the persona and simple FreeBASIC/poetry snippets. It does **not** truly reason
> or hold novel deep conversation, and it is **not** a mental-health service.
> For real distress, please reach out to a trusted person or a professional /
> crisis line — the model is trained to say the same.

## Architecture

```
tokens ─► token embedding + positional embedding
       ─► ┌─ LayerNorm ─► causal multi-head self-attention ─┐ + residual
          └──────────────────────────────────────────────── ┘
       ─► ┌─ LayerNorm ─► MLP (Linear ─► GELU ─► Linear) ────┐ + residual
          └──────────────────────────────────────────────── ┘   × NLAYER
       ─► final LayerNorm ─► linear head ─► softmax over the vocabulary
```

Defaults (all `Const`s at the top of `atlas.bas` — edit and rebuild to scale):

| setting  | value | meaning |
|----------|-------|---------|
| `BLOCK`  | 128   | context length (characters of history) |
| `NEMB`   | 192   | embedding / residual width |
| `NHEAD`  | 8     | attention heads |
| `NLAYER` | 6     | transformer blocks |
| `FFN`    | 768   | MLP hidden width |
| `NTHREAD`| 8     | worker threads (set near your CPU core count) |
| `BATCH`  | 16    | sequences per optimizer step |

≈ 2.7M parameters. Training uses AdamW (decoupled weight decay), global grad-norm
clipping, and a linear-warmup + cosine-decay learning-rate schedule.

### Multithreading

Training is data-parallel across `NTHREAD` threads: the mini-batch is split among
workers, each with its **own activation workspace and its own gradient buffers**
(so threads never write to shared memory), and the per-thread gradients are summed
after every step. On an 8-core CPU this is ~7× faster than single-threaded.

## Build

**Linux 64-bit**
```bash
./build_linux.sh          # fbc -gen gcc -O 3 -mt ... atlas.bas -x atlas
```

**Windows 32-bit** (32-bit FreeBASIC toolchain)
```bat
build_windows.bat         :: fbc -gen gcc -O 3 -mt ... atlas.bas -x atlas.exe
```

`-gen gcc -O 3` (fast inner loops) and **`-mt`** (multi-threaded runtime, required)
are both important.

## Use

```bash
./atlas train            # train ~2000 steps on data/corpus.txt, save model.bin
./atlas train 5000       # train for a specific number of steps (more = better)
./atlas chat             # load model.bin and talk to it
./atlas                  # chat if model.bin exists, otherwise train first
```

Training prints the loss (falls from ~3.5 toward ~1 and below) and every 500 steps
saves a checkpoint and prints a short sample, so `model.bin` is usable even if you
stop early.

### Chat

```
You: hello
Bot: well hello, my friend. come, sit with me a while.
You: i feel anxious
Bot: let us breathe together. slowly in, and slowly out. your worry is real, but it is not the whole truth of you.
You: recite a poem
Bot: here is a little verse for you: the evening folds its quiet wing, and in the dusk the sparrows sing...
You: how do i print in freebasic
Bot: you use print. for example: print "hello, world".
You: /quit
```

## The training data

`data/corpus.txt` is 1,000,000 lines of `User:` / `Bot:` dialogue in Atlas's
persona — greetings, life-wisdom reflections, mental-health support (validating,
and **crisis-aware**: distress is met with a caring nudge toward real help),
poetry, FreeBASIC Q&A, and encouragement. It is produced by
[`scripts/gen_corpus.py`](scripts/gen_corpus.py) from themed fragment pools:

```bash
python3 scripts/gen_corpus.py 1000000 data/corpus.txt
```

**Use your own data:** replace `data/corpus.txt` with any text (the `User: ...` /
`Bot: ...` line format is what the chat mode expects) and retrain. The vocabulary
is built automatically from whatever characters appear.

## How it works (tour of `atlas.bas`)

- **Parameter registry** — every weight tensor registered once; the optimizer,
  initializer, save/load, and per-thread gradient buffers all iterate over it.
- **`Workspace`** — all activations + backprop scratch for one sequence; one per
  worker thread.
- **`forward`** — embeddings → transformer blocks → final LayerNorm, storing every
  intermediate needed for backprop.
- **`backward`** — runs `forward`, computes softmax cross-entropy loss, walks the
  net in reverse accumulating gradients into this thread's buffers. Each op's
  gradient (LayerNorm, attention softmax, GELU, linears, embeddings) is derived
  inline in comments.
- **`workerProc` / `train`** — thread fan-out over the batch, gradient reduction,
  AdamW step, LR schedule, checkpointing, sampling.
- **`sampleNext`** — temperature sampling from the next-token distribution.

## Notes / limits

- Character-level + small model → short, simple, sometimes-imperfect output.
  Increasing `NEMB`/`NLAYER`/`BLOCK` and training longer improves coherence at the
  cost of speed (CPU-only).
- On this machine the defaults train at ~0.8 steps/s across 8 threads.
- `model.bin` stores the config + vocabulary; loading checks the compiled
  dimensions match.
- Reproducible by default (`Randomize 1234`); change that line for variety.
- **Not a substitute for professional help.** The mental-health content is a kind
  companion voice, nothing more.
```
