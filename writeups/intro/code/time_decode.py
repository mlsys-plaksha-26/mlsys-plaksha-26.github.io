#!/usr/bin/env python3
"""Measure batch-one BF16 decode, excluding prompt prefill and model loading.

Written by Codex (OpenAI).
"""

import argparse
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Hub ID or local path")
    parser.add_argument("--revision", default="main", help="Hub revision; use a commit for reproducibility")
    parser.add_argument("--device", type=int, default=0, help="visible CUDA device index")
    parser.add_argument("--prompt-tokens", type=positive_int, default=128)
    parser.add_argument("--decode-steps", type=positive_int, default=128,
                        help="cached forward passes AFTER the first token from prefill")
    parser.add_argument("--warmup-runs", type=positive_int, default=1)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--profile-step", type=positive_int,
                        help="profile just this 1-based cached decode step with CUDA profiler start/stop; skip timing runs")
    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="dense BF16 peak; defaults to 125 only on an NVIDIA A10")
    parser.add_argument("--output", type=Path,
                        help="JSON output (default: results/timing.json or results/profile-step-N.json)")
    args = parser.parse_args()
    if args.profile_step is not None and args.profile_step > args.decode_steps:
        parser.error("--profile-step must not exceed --decode-steps")
    if args.output is None:
        name = f"profile-step-{args.profile_step}" if args.profile_step else "timing"
        args.output = Path(f"results/{name}.json")
    if args.peak_tflops is not None and (
        not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0
    ):
        parser.error("--peak-tflops must be finite and positive")
    return args


def summary(values):
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def run_sequence(torch, model, prompt, steps):
    # Allocate/initialize events before timing, including their lazy CUDA handles.
    events = [torch.cuda.Event(enable_timing=True) for _ in range(steps + 1)]
    for event in events:
        event.record()
    torch.cuda.synchronize()

    prefill_start = time.perf_counter()
    output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del output
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_start) * 1000

    # No padding: the model constructs causal attention and cache positions.
    # Ignore EOS so every run has exactly the same number of decode steps.
    wall_start = time.perf_counter()
    events[0].record()
    for step in range(steps):
        output = model(input_ids=token, past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        events[step + 1].record()
        del output
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    token_ms = [events[i].elapsed_time(events[i + 1]) for i in range(steps)]
    return {"prefill_wall_ms": prefill_ms, "decode_wall_ms": wall_ms,
            "token_cuda_ms": token_ms}


def profile_step(torch, model, prompt, step):
    """Build a fresh cache, then expose exactly one decode step to Nsight Compute."""
    output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del output
    for _ in range(step - 1):
        output = model(input_ids=token, past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del output
    before = cache.get_seq_length()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        output = model(input_ids=token, past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    after = output.past_key_values.get_seq_length()
    assert before == prompt.shape[1] + step - 1 and after == before + 1
    return {"decode_step": step, "cache_tokens_before": before, "cache_tokens_after": after}


def main():
    args = parse_args()
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Run `uv sync` first.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check nvidia-smi and the CUDA PyTorch installation.")
    if not 0 <= args.device < torch.cuda.device_count():
        raise SystemExit(f"Invalid CUDA device index: {args.device}")
    torch.cuda.set_device(args.device)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The selected GPU does not support BF16.")
    device = torch.device("cuda", args.device)
    props = torch.cuda.get_device_properties(device)
    peak = args.peak_tflops
    if peak is None and props.name == "NVIDIA A10":
        peak = 125.0
    torch.manual_seed(0)

    print(f"GPU: {props.name} ({props.total_memory / 2**30:.2f} GiB)", flush=True)
    print(f"Loading {args.model} in BF16 on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map={"": str(device)}, attn_implementation=args.attention,
    ).eval()
    if getattr(model.config, "model_type", None) != "qwen3":
        raise SystemExit("This benchmark expects a dense Qwen3 model.")
    if any(p.device != device or p.dtype != torch.bfloat16 for p in model.parameters()):
        raise SystemExit("All model parameters must be resident on the selected GPU in BF16.")
    if args.prompt_tokens + args.decode_steps > model.config.max_position_embeddings:
        raise SystemExit("Prompt plus decode steps exceeds the model context length.")

    # The source gives a length but no prompt text. Repeat tokenized prose to
    # get exactly the requested number of real, unpadded input tokens.
    seed_ids = tokenizer.encode(
        "Explain why language model decoding depends on memory bandwidth and GPU compute. ",
        add_special_tokens=False,
    )
    ids = (seed_ids * math.ceil(args.prompt_tokens / len(seed_ids)))[:args.prompt_tokens]
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameters:,}; prompt: {prompt.shape[1]} tokens; "
          f"decode steps: {args.decode_steps}; batch: 1; attention: {args.attention}", flush=True)

    with torch.inference_mode():
        for i in range(args.warmup_runs):
            run_sequence(torch, model, prompt, args.profile_step or args.decode_steps)
            print(f"Warm-up {i + 1}/{args.warmup_runs} complete", flush=True)
        if args.profile_step is not None:
            profile = profile_step(torch, model, prompt, args.profile_step)
            report = {
                "profile": profile, "parameters": parameters,
                "approximate_flops_per_token": 2 * parameters,
                "settings": {**vars(args), "output": str(args.output), "batch_size": 1,
                             "dtype": "bfloat16", "prompt_ids": ids},
                "environment": {"gpu": props.name, "torch": torch.__version__,
                                "transformers": transformers.__version__, "cuda": torch.version.cuda,
                                "model_commit": getattr(model.config, "_commit_hash", None)},
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Profile range complete: {profile}. Metadata: {args.output}")
            return
        torch.cuda.reset_peak_memory_stats(device)
        runs = []
        for i in range(args.runs):
            result = run_sequence(torch, model, prompt, args.decode_steps)
            runs.append(result)
            print(f"Run {i + 1}/{args.runs}: "
                  f"CUDA {statistics.mean(result['token_cuda_ms']):.3f} ms/token, "
                  f"wall {result['decode_wall_ms'] / args.decode_steps:.3f} ms/token", flush=True)

    all_ms = [ms for run in runs for ms in run["token_cuda_ms"]]
    stats = summary(all_ms)
    wall_mean = sum(run["decode_wall_ms"] for run in runs) / len(all_ms)
    estimate_ms = 2 * parameters / (peak * 1e12) * 1000 if peak else None
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {**vars(args), "output": str(args.output), "batch_size": 1,
                     "dtype": "bfloat16", "cache": "dynamic", "greedy": True,
                     "ignore_eos": True, "prompt_ids": ids},
        "environment": {"gpu": props.name, "gpu_memory_bytes": props.total_memory,
                        "python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "cuda": torch.version.cuda,
                        "model_commit": getattr(model.config, "_commit_hash", None)},
        "parameters": parameters,
        "cuda_itl": {**stats, "tokens_per_second": 1000 / stats["mean_ms"]},
        "wall_decode": {"mean_ms_per_token": wall_mean, "tokens_per_second": 1000 / wall_mean},
        "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "compute_only": {"dense_bf16_peak_tflops": peak, "flops_per_token": 2 * parameters,
                         "estimate_ms": estimate_ms,
                         "cuda_measured_over_estimate": stats["mean_ms"] / estimate_ms if estimate_ms else None},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nMean CUDA ITL: {stats['mean_ms']:.3f} ms/token "
          f"({1000 / stats['mean_ms']:.2f} tokens/s)")
    print(f"Median / p95: {stats['median_ms']:.3f} / {stats['p95_ms']:.3f} ms")
    print(f"Wall decode: {wall_mean:.3f} ms/token ({1000 / wall_mean:.2f} tokens/s)")
    if estimate_ms is not None:
        print(f"Compute-only estimate (2P / {peak:g} TFLOP/s): {estimate_ms:.4f} ms/token")
        print(f"Measured CUDA / estimate: {stats['mean_ms'] / estimate_ms:.1f}x")
    print(f"Peak allocated GPU memory: {report['peak_memory_allocated_gib']:.2f} GiB")
    print(f"Saved raw timings and metadata: {args.output}")


if __name__ == "__main__":
    main()
