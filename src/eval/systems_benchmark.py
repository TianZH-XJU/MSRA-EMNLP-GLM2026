"""
Systems Benchmark v2: Prefill throughput + memory scaling for QC-MSA vs baselines.
Uses LLaMA 3.2-1B for 4k-8k (131K position support), GPT-2 for 512-1024.

Measures:
  1. Prefill throughput (tokens/sec) at 512..8192
  2. Peak GPU memory (GB)
  3. FA2 crossover point

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m src.eval.systems_benchmark
"""

import torch, gc, time, json, os, argparse, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from types import MethodType
from src.models.llama_forwards import make_llama_qcmsa_forward as _make_llama_qcmsa_forward, make_llama_local_forward as _make_llama_local_forward
import numpy as np


# ============================================================
# LLaMA forwards: canonical versions live in src/models/llama_forwards.py (imported below)

# Benchmark harness
# ============================================================

def benchmark_model(model, seq_lengths, device, warmup=3, repeats=5, label="", model_type=""):
    """Benchmark prefill throughput and peak memory."""
    results = []
    batch_sizes = [int(b) for b in os.environ.get("SYSTEMS_BATCH_SIZES", "1").split(",")]
    # bs-major order: batch=8 OOMs fragment the heap and poison later batch=1 rows
    for bs in batch_sizes:
      for seq_len in seq_lengths:
        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        input_ids = torch.randint(0, 10000, (bs, seq_len), device=device)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)

        try:
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(input_ids, position_ids=pos_ids, use_cache=False)
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(repeats):
                    _ = model(input_ids, position_ids=pos_ids, use_cache=False)
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:>20} @{seq_len:>5}: OOM")
            results.append({'seq_len': seq_len, 'label': label, 'model_type': model_type,
                            'tokens_per_sec': 0, 'ms_per_run': 0, 'peak_memory_gb': 0, 'status': 'OOM'})
            continue
        except Exception as e:
            print(f"  {label:>20} @{seq_len:>5}: ERROR — {str(e)[:80]}")
            results.append({'seq_len': seq_len, 'label': label, 'model_type': model_type,
                            'tokens_per_sec': 0, 'ms_per_run': 0, 'peak_memory_gb': 0, 'status': str(e)[:80]})
            continue

        tokens_per_sec = (seq_len * repeats * bs) / elapsed
        ms_per_run = (elapsed / repeats) * 1000
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

        results.append({
            'seq_len': seq_len, 'tokens_per_sec': round(tokens_per_sec, 1),
            'ms_per_run': round(ms_per_run, 1), 'peak_memory_gb': round(peak_mem, 2),
            'label': label, 'model_type': model_type, 'status': 'ok',
        })
        print(f"  {label:>20} @{seq_len:>5}: {tokens_per_sec:>8.0f} tok/s, "
              f"{ms_per_run:>7.1f} ms, {peak_mem:>5.2f} GB")

    return results


@torch.no_grad()
def benchmark_decode(model, seq_len, new_tokens, device, warmup=2, label="", batch_size=1):
    """③ autoregressive decode throughput: prefill once (KV cache), then step new_tokens.
    Expect ~local speed (far branch is prefill-only) — the paper's practical good-news row."""
    results = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # release prefill-time FFT buffer caches: they would otherwise sit in GPU memory
    # during decode (measured: 21.7 GB resident -> decode 56 vs 87 tok/s for local)
    try:
        for layer in model.model.layers:
            sp = getattr(layer.self_attn, 'qcmsa_spectral', None)
            if sp is not None:
                sp.clear_caches()
    except AttributeError:
        pass
    try:
        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        out = model(input_ids, position_ids=pos_ids, use_cache=True)
        past, cur, pos = out.past_key_values, input_ids[:, -1:], seq_len
        for _ in range(3):
            out = model(cur, position_ids=torch.full((batch_size, 1), pos, device=device, dtype=torch.long),
                        past_key_values=past, use_cache=True)
            past, cur, pos = out.past_key_values, out.logits[:, -1:].argmax(-1), pos + 1
        torch.cuda.synchronize()
        import time as _t
        # prefill is UNTIMED (the old mixed-prefill bug diluted qcmsa decode by its 0.5s prefill)
        out = model(input_ids, position_ids=pos_ids, use_cache=True)
        past, cur, pos = out.past_key_values, input_ids[:, -1:], seq_len
        torch.cuda.synchronize()
        t0 = _t.perf_counter()
        for _ in range(new_tokens):
            out = model(cur, position_ids=torch.full((batch_size, 1), pos, device=device, dtype=torch.long),
                        past_key_values=past, use_cache=True)
            past, cur, pos = out.past_key_values, out.logits[:, -1:].argmax(-1), pos + 1
        torch.cuda.synchronize()
        elapsed = _t.perf_counter() - t0
        tok_s = (new_tokens * batch_size) / elapsed
        mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        results.append({'seq_len': seq_len, 'new_tokens': new_tokens, 'batch_size': batch_size,
                        'decode_tok_per_sec': round(tok_s, 1),
                        'ms_per_token': round(elapsed / (new_tokens * batch_size) * 1000, 3),
                        'peak_memory_gb': round(mem, 2), 'label': label, 'status': 'ok'})
        print(f"  {label:>20} decode @{seq_len:>5} +{new_tokens}: {tok_s:8.0f} tok/s, {mem:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:>20} decode @{seq_len:>5}: OOM")
        results.append({'seq_len': seq_len, 'new_tokens': new_tokens, 'batch_size': batch_size,
                        'decode_tok_per_sec': 0, 'label': label, 'status': 'OOM'})
    return results


@torch.no_grad()
def profile_prefill(model, seq_len, device, label=""):
    """③ one-line op decomposition supporting the engineering-headroom claim."""
    from torch.profiler import profile, ProfilerActivity
    input_ids = torch.randint(0, 10000, (1, seq_len), device=device)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    for _ in range(2):
        _ = model(input_ids, position_ids=pos_ids, use_cache=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _ = model(input_ids, position_ids=pos_ids, use_cache=False)
        torch.cuda.synchronize()
    rows = [{"op": e.key[:60], "self_cuda_ms": round(e.self_device_time_total / 1000, 2)}
            for e in prof.key_averages() if e.self_device_time_total > 0]
    rows.sort(key=lambda r: -r["self_cuda_ms"])
    print(f"  {label:>20} profiler @{seq_len}: " + ", ".join(f"{r['op'][:24]}={r['self_cuda_ms']}ms" for r in rows[:4]))
    return rows[:10]


def load_llama_full(model_path, device, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    return model


def load_llama_local(model_path, device, window_size=128, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(
            _make_llama_local_forward(window_size), layer.self_attn)
    return model


def load_llama_qcmsa(model_path, device, window_size=128, patch_layers=8,
                     feature_rank=16, num_experts=4, use_gate=True, dtype=torch.float16):
    from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager").to(device).eval()

    n_layer = len(model.model.layers)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    start_layer = n_layer - min(patch_layers, n_layer)

    kcfg = QCMSAConfig(
        head_dim=head_dim, feature_rank=feature_rank, num_experts=num_experts,
        kernel_types=tuple(["exponential"] * num_experts),
        use_gate=use_gate, window_size=window_size, dtype=torch.float32,
    )

    for i, layer in enumerate(model.model.layers):
        if i >= start_layer:
            sp = CorrectSpectralAttention(kcfg).to(device=device, dtype=torch.float32)
            sp.log_sigma.data = torch.tensor(
                [3.0, 4.0, 5.0, 6.0][:num_experts], device=device)
            layer.self_attn.forward = MethodType(
                _make_llama_qcmsa_forward(sp, window_size), layer.self_attn)
        else:
            layer.self_attn.forward = MethodType(
                _make_llama_local_forward(window_size), layer.self_attn)

    return model


# ============================================================
# Main
# ============================================================

# thread cap: torch defaults to 128 OMP threads on this 256-core host; concurrent
# background jobs then oversubscribe CPU to >5000% (2026-09-07 ops incident)
torch.set_num_threads(8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama",
                        choices=["gpt2", "llama"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/project/freq-attention/results")
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--patch_layers", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--full", action="store_true",
                        help="Include full attention")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode_tokens", type=int, default=64,
                        help="③ autoregressive decode tokens to measure (0 = skip)")
    parser.add_argument("--batch_sizes", type=str, default="1",
                        help="③ comma list of prefill batch sizes, e.g. 1,8")
    parser.add_argument("--with_profile", action="store_true", help="③ one-line op decomposition")
    parser.add_argument("--suffix", type=str, default="", help="output filename suffix (v2 protocol: never overwrite old results)")
    parser.add_argument("--num_experts", type=int, default=4, help="E5: spectral expert count M (throughput depends on structure, not trained values)")
    parser.add_argument("--gate_off", action="store_true", help="E5: disable gating (matches M1L16 default config)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Model config
    if args.model == "gpt2":
        if args.model_path is None:
            args.model_path = "/data1/zhoujun/Auto-claude-code-research-in-sleep/basemodel/gpt2"
        # GPT-2 caps at 1024 due to position embedding
        seq_lengths = [512, 1024]
        model_family = "gpt2"
    else:
        if args.model_path is None:
            args.model_path = "/data1/zhoujun/Auto-claude-code-research-in-sleep/basemodel/llama3.2-1b"
        seq_lengths = [512, 1024, 2048, 4096]
        if args.max_len >= 8192:
            seq_lengths.append(8192)

    if args.repeats == 1:
        warmup = 1
    else:
        warmup = min(3, args.repeats)

    print("=" * 70)
    print(f"QC-MSA SYSTEMS BENCHMARK — {args.model.upper()}")
    print(f"Model: {args.model_path}")
    print(f"Window={args.window_size}, PatchLayers={args.patch_layers}")
    print(f"Seq lengths: {seq_lengths}, Full: {args.full}")
    print("=" * 70)

    all_results = {}

    # 1. Full Attention
    if args.full:
        print("\n--- 1. Full Attention ---")
        gc.collect(); torch.cuda.empty_cache()
        if args.model == "llama":
            model = load_llama_full(args.model_path, device)
        else:
            from src.eval.safe_loader import load_full
            model = load_full(args.model_path).to(device).eval()
        r = benchmark_model(model, seq_lengths, device, warmup, args.repeats, "Full", args.model)
        all_results["full"] = r
        if args.decode_tokens > 0 and args.model == "llama":
            all_results.setdefault("full_decode", []).extend(
                benchmark_decode(model, 2048, args.decode_tokens, device, label="full"))
        if args.with_profile and args.model == "llama":
            all_results.setdefault("full_profile", []).extend(
                profile_prefill(model, 2048, device, label="full"))
        del model; gc.collect(); torch.cuda.empty_cache()

    # 2. Local (ws=128)
    print("\n--- 2. Local (custom mask, ws=128) ---")
    gc.collect(); torch.cuda.empty_cache()
    if args.model == "llama":
        model = load_llama_local(args.model_path, device, args.window_size)
    else:
        from src.eval.safe_loader import load_local
        model = load_local(args.model_path, window_size=args.window_size).to(device).eval()
    r = benchmark_model(model, seq_lengths, device, warmup, args.repeats, "Local ws=128", args.model)
    all_results["local"] = r
    if args.decode_tokens > 0 and args.model == "llama":
        all_results.setdefault("local_decode", []).extend(
            benchmark_decode(model, 2048, args.decode_tokens, device, label="local"))
    if args.with_profile and args.model == "llama":
        all_results.setdefault("local_profile", []).extend(
            profile_prefill(model, 2048, device, label="local"))
    del model; gc.collect(); torch.cuda.empty_cache()

    # 3. QC-MSA
    print("\n--- 3. QC-MSA (local + far branch) ---")
    gc.collect(); torch.cuda.empty_cache()
    if args.model == "llama":
        model = load_llama_qcmsa(args.model_path, device, args.window_size, args.patch_layers,
                                 num_experts=args.num_experts, use_gate=not args.gate_off)
    else:
        from src.eval.safe_loader import load_qcmsa
        model = load_qcmsa(args.model_path, window_size=args.window_size,
                           patch_last_n=args.patch_layers).to(device).eval()
    # E-B: decode BEFORE the prefill loop — the @4096/@8192 OOM rows fragment the heap
    # and would push the (otherwise-fine, ~15GB) decode phase over 24GB on a fresh cache
    if args.decode_tokens > 0 and args.model == "llama":
        all_results.setdefault("qcmsa_decode", []).extend(
            benchmark_decode(model, 2048, args.decode_tokens, device, label="qcmsa"))
    if args.with_profile and args.model == "llama":
        all_results.setdefault("qcmsa_profile", []).extend(
            profile_prefill(model, 2048, device, label="qcmsa"))
    r = benchmark_model(model, seq_lengths, device, warmup, args.repeats, "QC-MSA ws=128", args.model)
    all_results["qcmsa"] = r
    del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    out_file = os.path.join(args.output_dir, f"systems_{args.model}{args.suffix}.json")
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Summary
    labels_order = ["local", "qcmsa"]
    if args.full:
        labels_order.insert(0, "full")

    print("\n" + "=" * 70)
    print("THROUGHPUT (tokens/sec)")
    print("=" * 70)
    header = f"{'Seq':<6}"
    for lb in labels_order:
        header += f" {lb:>16}"
    print(header)
    print("-" * len(header))
    for i, L in enumerate(seq_lengths):
        row = f"{L:<6}"
        for lb in labels_order:
            entries = all_results.get(lb, [])
            if i < len(entries) and entries[i].get('status') == 'ok':
                row += f" {entries[i]['tokens_per_sec']:>16.0f}"
            else:
                row += f" {'—':>16}"
        print(row)

    print("\n" + "=" * 70)
    print("PEAK MEMORY (GB)")
    print("=" * 70)
    header = f"{'Seq':<6}"
    for lb in labels_order:
        header += f" {lb:>16}"
    print(header)
    print("-" * len(header))
    for i, L in enumerate(seq_lengths):
        row = f"{L:<6}"
        for lb in labels_order:
            entries = all_results.get(lb, [])
            if i < len(entries) and entries[i].get('status') == 'ok':
                row += f" {entries[i]['peak_memory_gb']:>16.2f}"
            else:
                row += f" {'—':>16}"
        print(row)

    # Overhead
    if all_results.get("local") and all_results.get("qcmsa"):
        print("\n--- QC-MSA Overhead vs Local ---")
        for i, L in enumerate(seq_lengths):
            lt = all_results["local"][i].get("tokens_per_sec", 0)
            qt = all_results["qcmsa"][i].get("tokens_per_sec", 0)
            if lt > 0 and qt > 0:
                oh = (lt - qt) / lt * 100
                print(f"  @{L}: {oh:+.1f}% ({lt:.0f} → {qt:.0f} tok/s)")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
