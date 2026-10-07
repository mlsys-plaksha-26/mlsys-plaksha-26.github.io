# Qwen3-8B decode latency on an A10

**The benchmark code in `time_decode.py` was written by Codex (OpenAI).**
This folder includes the script, its `pyproject.toml`, and the existing `uv.lock`.

## Run

Use a Linux machine with an NVIDIA A10 (24 GB GDDR6), a compatible NVIDIA
driver, Python 3.12, and uv. Run from this folder with the GPU otherwise idle.
The original benchmark command was:

```sh
uv run time_decode.py
```

To enforce the supplied dependency lockfile and the recorded model revision:

```sh
uv run --locked --python 3.12 time_decode.py --revision b968826d9c46dd6066d109eabc6255188de91218
```

The first run installs dependencies and downloads model weights if they are not
already cached. The script defaults to the model's `main` revision unless
`--revision` is supplied.

## Workload and timing

Defaults: Qwen/Qwen3-8B, batch size 1, BF16 weights and model computation, SDPA
attention, and a dynamic KV cache. The script repeats tokenized prose to make
exactly 128 prompt tokens. One warm-up sequence precedes three measured
sequences, each with 128 cached decode steps (context lengths 129 through 256).
GPU greedy argmax is included; EOS is ignored. Normal FP32 normalization and
accumulation within BF16 inference remain enabled.

CUDA events measure successive decode boundaries, with synchronization after
the loop. Synchronized wall-clock timing is also reported. Both exclude model
loading, tokenization, warm-up, and prefill. The first generated token comes from
prefill and is excluded from the 128 measured decode steps. Throughput is for
decode only, without CPU token-text streaming.

## Reference result

The saved A10 run recorded **44.966 ms/token (22.239 tokens/s)** using CUDA
events, and **44.967 ms/token** using wall-clock timing: approximately
**45.0 ms/token and 22.2 tokens/s**.

Recorded environment: Python 3.12.12, PyTorch 2.7.1+cu126, Transformers 4.57.1,
CUDA runtime 12.6, and the model revision in the command above. The NVIDIA
driver version was not recorded in that run. Results can vary with GPU load,
clocks, driver, and software environment.

The script saves settings, environment, model commit, and raw timings to
`results/timing.json` (overwritten on subsequent runs).

Its printed compute-only comparison uses the rough `2P` estimate over **all**
model parameters: about 16.38 GFLOPs/token and 0.131 ms at 125 TFLOPs/s.
The lecture's more precise estimate is about **15.2 GFLOPs/token**, accounting
for the matrices actually used and short-context attention; input embeddings
are looked up rather than fully multiplied. This packaging preserves the
benchmark's timing behavior and its original compute-only comparison.
