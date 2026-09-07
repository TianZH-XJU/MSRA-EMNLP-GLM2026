"""
NIAH Distance Grid — decisive far-context recovery evidence.

Tests whether QC-MSA can retrieve a "needle" fact embedded at controlled
distances in a natural-text haystack, where local attention fails at far distances.

Uses natural text (PG-19) as haystack with a synthetic fact inserted.
Measures answer NLL (lower = better retrieval).

Supports: GPT-2 (via safe_loader) and LLaMA 3.2-1B.
Multiple seq_lens, window_sizes, and models.

Usage:
  CUDA_VISIBLE_DEVICES=4 python -m src.eval.niah_grid \
    --model_family llama --seq_lens 1024,2048,4096 --window_sizes 128,256 \
    --models local,qcmsa --ckpt results/llama_512_s2.pt --num_examples 30

Output: JSON per config with per-example results → plot with src/eval/plot_niah.py
"""

import torch, torch.nn.functional as F, gc, random, json, os, argparse, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm


# ============================================================
# Haystack generation
# ============================================================

def load_haystack_texts(pg19_path, max_chars=200000):
    """Load PG-19 texts for natural haystacks."""
    import pandas as pd
    data_dir = os.path.join(pg19_path, "data")
    parquet_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")])
    dfs = [pd.read_parquet(pf) for pf in parquet_files[:2]]
    all_df = pd.concat(dfs, ignore_index=True)
    texts = []
    for t in all_df["text"].tolist():
        if len(t) > 1000:
            texts.append(t)
        if sum(len(t) for t in texts) > max_chars:
            break
    return texts


def generate_niah_example(seq_len, evidence_pos, tokenizer, haystack_texts, seed=0):
    """Generate a NIAH example with natural haystack and a copy-based needle.

    Uses a simple repetition format that base LMs can do:
    "The secret key is ZEBRA789. Remember ZEBRA789. [...] The secret key is [ZEBRA789]"

    The model needs to attend to the earlier occurrence of ZEBRA789 to predict it.
    """
    rng = random.Random(seed)

    # Unique key words that tokenize consistently
    key_words = ["ZE", "BRA"]  # tokens for "ZEBRA"
    num_str = str(rng.randint(100, 999))
    num_tokens = tokenizer.encode(num_str, add_special_tokens=False)

    # Build haystack from natural text
    haystack_tokens = []
    for text in haystack_texts:
        toks = tokenizer.encode(text, add_special_tokens=False)
        haystack_tokens.extend(toks)
        if len(haystack_tokens) >= seq_len + 500:
            break
    haystack_tokens = haystack_tokens[:seq_len]

    # Needle: a distinctive rare token sequence (not in normal English)
    # "The secret key is XYZZY-{num}. Remember XYZZY-{num}."
    needle_prefix = tokenizer.encode(" The secret key is ", add_special_tokens=False)
    # Use a distinctive marker token
    marker = tokenizer.encode("XYZZY", add_special_tokens=False)
    dash = tokenizer.encode("-", add_special_tokens=False)

    needle_ids = needle_prefix + marker + dash + num_tokens
    # Small reinforcement
    remember = tokenizer.encode(" Remember XYZZY-", add_special_tokens=False)
    needle_ids = needle_ids + remember + num_tokens + tokenizer.encode(".", add_special_tokens=False)

    # Target: what the model should predict = " XYZZY-{num}"
    target_prefix = tokenizer.encode(" XYZZY-", add_special_tokens=False)
    answer_ids = target_prefix + num_tokens

    # Query at the end: "The secret key is"
    query_ids = tokenizer.encode(" The secret key is", add_special_tokens=False)

    # E-C fix (P0-2): truncate to the FINAL haystack length BEFORE inserting the
    # needle. The pre-fix order (insert into the seq_len haystack, then re-truncate
    # to max_haystack) silently cut the needle tail — including the retrieval target
    # — for near-bucket distances, seeding the bucket with unanswerable samples.
    max_haystack = seq_len - len(query_ids) - len(answer_ids) - 5
    haystack_tokens = haystack_tokens[:max_haystack]

    # Place needle at evidence_pos, clamped so the FULL needle fits the final
    # haystack (shifting left only increases the actual distance, keeping near
    # samples in-window: actual_distance <= len(needle) + 2*answer_len + slack)
    pos = min(evidence_pos, max(0, max_haystack - len(needle_ids)))
    haystack_tokens[pos:pos+len(needle_ids)] = needle_ids

    # E-C: hard survival assertion — the placed needle must be retained verbatim
    placed = haystack_tokens[pos:pos+len(needle_ids)]
    assert placed == needle_ids, (
        f"needle truncated after placement: pos={pos}, needle_len={len(needle_ids)}, "
        f"max_haystack={max_haystack} (audit A.9)")

    input_ids = torch.tensor(haystack_tokens + query_ids).unsqueeze(0)
    query_start = len(haystack_tokens)
    actual_distance = query_start - (pos + len(needle_ids))

    return {
        'input_ids': input_ids,
        'evidence_pos': pos,
        'query_start': query_start,
        'target_ids': torch.tensor(answer_ids),
        'needle_value': f"XYZZY-{num_str}",
        'total_len': input_ids.shape[1],
        'needle_survived': True,
        'actual_distance': actual_distance,
    }


# ============================================================
# Model loading
# ============================================================

def load_model(model_family, model_path, model_type, window_size, device, ckpt_path=None):
    """Load model with proper configuration for NIAH eval."""
    from src.eval.safe_loader import load_full, load_local, load_qcmsa

    if model_family == "gpt2":
        if model_type == "full":
            model = load_full(model_path).to(device).eval()
        elif model_type == "local":
            model = load_local(model_path, window_size=window_size).to(device).eval()
        elif model_type == "qcmsa":
            model = load_qcmsa(model_path, window_size=window_size, patch_last_n=12)
            model.to(device).eval()
            if ckpt_path and os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                n_layer = model.config.n_layer
                for i, blk in enumerate(model.transformer.h):
                    if hasattr(blk.attn, 'qcmsa_spectral') and blk.attn.qcmsa_spectral is not None:
                        key = str(i)
                        if key in ckpt:
                            sd = {k: v.to(device) for k, v in ckpt[key].items()}
                            blk.attn.qcmsa_spectral.load_state_dict(sd, strict=False)
                            blk.attn.qcmsa_spectral.clear_caches()
    else:  # llama
        if model_type == "full":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16).to(device).eval()
        elif model_type == "local":
            from transformers import AutoModelForCausalLM
            from types import MethodType
            from src.models.llama_forwards import make_llama_local_forward as _make_llama_local_forward
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16,
                                                          attn_implementation="eager").to(device).eval()
            for layer in model.model.layers:
                layer.self_attn.forward = MethodType(_make_llama_local_forward(window_size), layer.self_attn)
        elif model_type == "qcmsa":
            from transformers import AutoModelForCausalLM
            from types import MethodType
            from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig
            from src.models.llama_forwards import make_llama_local_forward as _make_llama_local_forward, make_llama_qcmsa_forward as _make_llama_qcmsa_forward

            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16,
                                                          attn_implementation="eager").to(device).eval()
            n_layer = len(model.model.layers)
            patch_n = min(8, n_layer)
            start_layer = n_layer - patch_n
            head_dim = model.config.hidden_size // model.config.num_attention_heads

            # infer M / gate / patch depth from the checkpoint (M=1 and M=4 ckpts both occur)
            ckpt = torch.load(ckpt_path, map_location=device) if ckpt_path and os.path.exists(ckpt_path) else None
            if ckpt is not None:
                sd0 = next(iter(ckpt.values()))
                n_exp = len(sd0["log_sigma"])
                use_gate = "gate.weight" in sd0
                start_layer = n_layer - min(len(ckpt), n_layer)
                patch_n = n_layer - start_layer
            else:
                n_exp, use_gate = 4, True
            kcfg = QCMSAConfig(head_dim=head_dim, feature_rank=16, num_experts=n_exp,
                               kernel_types=("exponential",)*n_exp, use_gate=use_gate,
                               window_size=window_size, dtype=torch.float32)

            for i, layer in enumerate(model.model.layers):
                if i >= start_layer:
                    sp = CorrectSpectralAttention(kcfg).to(device=device, dtype=torch.float32)
                    if ckpt is None:
                        sp.log_sigma.data = torch.tensor([3.0, 4.0, 5.0, 6.0][:n_exp], device=device)
                    layer.self_attn.qcmsa_spectral = sp
                    layer.self_attn.forward = MethodType(_make_llama_qcmsa_forward(sp, window_size), layer.self_attn)
                else:
                    layer.self_attn.forward = MethodType(_make_llama_local_forward(window_size), layer.self_attn)

            if ckpt is not None:
                for i, layer in enumerate(model.model.layers):
                    if hasattr(layer.self_attn, 'qcmsa_spectral') and layer.self_attn.qcmsa_spectral is not None:
                        key = str(i)
                        if key in ckpt:
                            sd = {k: v.to(device) for k, v in ckpt[key].items()}
                            layer.self_attn.qcmsa_spectral.load_state_dict(sd, strict=False)
                            layer.self_attn.qcmsa_spectral.clear_caches()

    return model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_niah_model(model, tokenizer, examples, device, label=""):
    """Evaluate NIAH retrieval loss for a set of examples."""
    results = []
    total_loss, total_tokens = 0.0, 0

    for ex in tqdm(examples, desc=label, leave=False):
        input_ids = ex['input_ids'].to(device)
        target_ids = ex['target_ids'].to(device)
        query_start = ex['query_start']

        try:
            # F0 fix (expert review): the answer must be IN the input and the loss must
            # sit on the answer region. The old code scored the query phrase's own
            # continuation (target_ids never used) — pure local LM, retrieval-blind.
            full_ids = torch.cat([input_ids, target_ids.unsqueeze(0)], dim=1)
            p_len = input_ids.shape[1]
            out = model(full_ids)
            logits = out.logits[0, p_len-1:-1, :]          # predict answer tokens
            target = full_ids[0, p_len:]                   # the answer
            if len(target) > 0 and len(logits) > 0:
                min_len = min(len(logits), len(target))
                loss = F.cross_entropy(logits[:min_len], target[:min_len], reduction='sum')
                n_tokens = min_len
                total_loss += loss.item()
                total_tokens += n_tokens
                avg_loss = loss.item() / n_tokens
            else:
                avg_loss = float('nan')
        except Exception as e:
            avg_loss = float('nan')
            n_tokens = 0

        # Use pre-computed query-relative distance from example, or compute it
        distance = ex.get('distance', max(0, query_start - ex['evidence_pos']))
        results.append({
            'evidence_pos': ex['evidence_pos'],
            'query_start': query_start,
            'distance': distance,
            'avg_loss': avg_loss,
            'total_len': ex['total_len'],
        })

    overall_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
    return results, overall_loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", type=str, default="llama", choices=["gpt2", "llama"])
    parser.add_argument("--model_path", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/basemodel/llama3.2-1b")
    parser.add_argument("--pg19_path", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/dataset/pg19")
    parser.add_argument("--seq_lens", type=str, default="1024,2048,4096")
    parser.add_argument("--window_sizes", type=str, default="128,256")
    parser.add_argument("--models", type=str, default="local,qcmsa")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num_examples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/project/freq-attention/results/niah")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    seq_lengths = [int(s) for s in args.seq_lens.split(",")]
    window_sizes = [int(w) for w in args.window_sizes.split(",")]
    model_types = args.models.split(",")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    haystack_texts = load_haystack_texts(args.pg19_path)

    print("=" * 70)
    print(f"NIAH DISTANCE GRID — {args.model_family.upper()}")
    print(f"Seq lengths: {seq_lengths}, Windows: {window_sizes}")
    print(f"Models: {model_types}, Examples per config: {args.num_examples}")
    print("=" * 70)

    all_results = {}

    for ws in window_sizes:
        all_results[f"ws{ws}"] = {}

        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"ws={ws}, model={model_type}")
            print(f"{'='*60}")

            gc.collect(); torch.cuda.empty_cache()
            model = load_model(args.model_family, args.model_path, model_type, ws, device, args.ckpt)

            for seq_len in seq_lengths:
                # Distance is query-to-evidence (tokens before query).
                # Query is at the end, so evidence distance ≈ seq_len - evidence_pos.
                # Bucket by distance from query:
                #   near: ≤ws (within local window)
                #   mid:  ws+1 to 2*ws
                #   far:  >2*ws
                # Evidence position is measured from the end: we place evidence
                # at query_start - distance.
                # query_start is approximately seq_len - len(query_ids).

                # Pre-compute approximate query position
                query_len = len(tokenizer.encode(" The secret key is", add_special_tokens=False))
                approx_query_start = seq_len - query_len - 5  # reserve space for answer

                distances = [
                    (1, ws, "near"),                          # within window
                    (ws + 1, min(2 * ws, seq_len - 20), "mid"),  # slightly beyond
                    (2 * ws + 1, seq_len - 20, "far"),         # far beyond
                ]

                # Generate examples
                examples = []
                for d_idx, (d_min, d_max, d_label) in enumerate(distances):
                    if d_min > d_max or d_min >= approx_query_start:
                        continue
                    n_d = args.num_examples // len(distances) + 1
                    for ex_i in range(n_d):
                        # Place evidence at query_start - distance
                        dist = random.randint(d_min, min(d_max, approx_query_start - 1))
                        ev_pos = approx_query_start - dist
                        seed = args.seed * 1000 + d_idx * 100 + ex_i
                        ex = generate_niah_example(seq_len, ev_pos, tokenizer,
                                                    haystack_texts, seed=seed)
                        ex['distance_bucket'] = d_label
                        ex['distance'] = dist  # Override with query-relative distance
                        examples.append(ex)

                print(f"  @seq={seq_len}: {len(examples)} examples")

                results, overall_loss = evaluate_niah_model(model, tokenizer, examples, device,
                                                             f"ws{ws}/{model_type}/{seq_len}")
                print(f"    Overall NLL: {overall_loss:.3f}")

                # E-C (P0-2): survival is asserted per example at generation time;
                # the aggregate rate is re-checked here and recorded in the output
                n_survived = sum(1 for ex in examples if ex.get('needle_survived'))
                survival_rate = n_survived / len(examples) if examples else 0.0
                assert survival_rate == 1.0, (
                    f"needle survival rate {survival_rate:.3f} < 1.0 (audit A.9)")

                # Aggregate by distance bucket
                buckets = {}
                for r, ex in zip(results, examples):
                    bucket = ex['distance_bucket']
                    if bucket not in buckets:
                        buckets[bucket] = {'losses': [], 'distances': []}
                    if not np.isnan(r['avg_loss']):
                        buckets[bucket]['losses'].append(r['avg_loss'])
                        buckets[bucket]['distances'].append(r['distance'])

                bucket_summary = {}
                for bucket, data in buckets.items():
                    if data['losses']:
                        bucket_summary[bucket] = {
                            'mean_loss': float(np.mean(data['losses'])),
                            'std_loss': float(np.std(data['losses'])),
                            'n': len(data['losses']),
                            'mean_distance': float(np.mean(data['distances'])),
                        }

                key = f"{model_type}_{seq_len}"
                all_results[f"ws{ws}"][key] = {
                    'model': model_type, 'window': ws, 'seq_len': seq_len,
                    'overall_nll': overall_loss,
                    'needle_survival_rate': survival_rate,
                    'buckets': bucket_summary,
                    'per_example': results,
                }

                # Print bucket summary
                for bucket in ["near", "mid", "far"]:
                    if bucket in bucket_summary:
                        bs = bucket_summary[bucket]
                        print(f"    {bucket:>6}: NLL={bs['mean_loss']:.3f}±{bs['std_loss']:.3f} (n={bs['n']})")

            del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    out_file = os.path.join(args.output_dir, f"niah_{args.model_family}.json")
    json.dump(all_results, open(out_file, 'w'), indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print("NIAH SUMMARY: NLL by distance bucket (lower = better retrieval)")
    print(f"{'='*80}")

    for ws in window_sizes:
        for model_type in model_types:
            print(f"\n--- ws={ws}, {model_type} ---")
            header = f"{'Seq':<8}"
            for bucket in ["near", "mid", "far"]:
                header += f" {bucket:>14}"
            print(header)
            print("-" * len(header))
            for seq_len in seq_lengths:
                key = f"{model_type}_{seq_len}"
                row = f"{seq_len:<8}"
                for bucket in ["near", "mid", "far"]:
                    bs = all_results.get(f"ws{ws}", {}).get(key, {}).get("buckets", {}).get(bucket, {})
                    if bs:
                        row += f" {bs['mean_loss']:>14.3f}"
                    else:
                        row += f" {'—':>14}"
                print(row)

    # Far-recovery delta
    print(f"\n--- Far-Recovery Delta (QC-MSA - Local) ---")
    for ws in window_sizes:
        for seq_len in seq_lengths:
            local_far = all_results.get(f"ws{ws}", {}).get(f"local_{seq_len}", {}).get("buckets", {}).get("far", {})
            qcmsa_far = all_results.get(f"ws{ws}", {}).get(f"qcmsa_{seq_len}", {}).get("buckets", {}).get("far", {})
            if local_far and qcmsa_far:
                delta = local_far['mean_loss'] - qcmsa_far['mean_loss']
                direction = "QC-MSA BETTER" if delta > 0 else "local better"
                print(f"  ws={ws} @{seq_len}: local={local_far['mean_loss']:.3f}, "
                      f"qcmsa={qcmsa_far['mean_loss']:.3f}, Δ={delta:+.3f} ({direction})")

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
