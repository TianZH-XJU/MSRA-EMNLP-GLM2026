"""
RULER minimum viable evaluation (needle-aware truncation to a 2,048-token eval window).

RULER original is 128K — too long for our setup. We truncate to 2,048 tokens at eval time
and evaluate using PPL-based scoring (same as LongBench).

Supports: niah_single, niah_multikey, qa_squad, qa_hotpotqa

Usage:
  CUDA_VISIBLE_DEVICES=7 python -m src.eval.ruler_eval --task niah_single --max_samples 30
"""

import torch, torch.nn.functional as F, gc, json, os, argparse, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.eval.llama_loader import load_llama_full, load_llama_local
from transformers import AutoTokenizer
from tqdm import tqdm


def load_ruler_examples(data_dir, task, max_samples=30, max_tokens=4096, needle_depth=0.5):
    """⑥ needle-aware loading: keep a window that CENTERS the needle at `needle_depth`
    of the context, then assert survival. The old head-truncation dropped the needle in
    0/30 checked examples — those runs measured needle-less answer PPL (invalid)."""
    task_paths = {
        'niah_single': 'niah/niah_single_1.jsonl',
        'niah_multikey': 'niah/niah_multikey_1.jsonl',
        'qa_squad': 'qa/squad.jsonl',
        'qa_hotpotqa': 'qa/hotpotqa.jsonl',
    }
    if task not in task_paths:
        raise ValueError(f"Unknown RULER task: {task}. Available: {list(task_paths.keys())}")

    path = os.path.join(data_dir, task_paths[task])
    if not os.path.exists(path):
        raise FileNotFoundError(f"RULER task not found: {path}")

    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            examples.append({
                'context': d.get('context', ''),
                'query': d.get('query', ''),
                'outputs': d.get('outputs', []),
            })
            if len(examples) >= max_samples:
                break
    return examples


def needle_aware_truncate(context, needle, tokenizer, max_ctx_tokens, needle_depth=0.5):
    """Center the needle at `needle_depth` of a max_ctx_tokens window; assert it survives.
    Returns (window_text, survived_bool). The needle is the answer/evidence string."""
    if not needle:
        raise ValueError("empty needle — cannot assert survival")
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    n_needle = len(needle_ids)
    if n_needle >= max_ctx_tokens:
        raise ValueError("needle longer than context window")

    # locate first occurrence of the needle (token-level search via char offset fallback)
    ctx_text_chars = context
    pos_chars = ctx_text_chars.find(needle)
    if pos_chars < 0:
        return tokenizer.decode(ctx_ids[:max_ctx_tokens]), False  # needle absent in source
    prefix_ids = tokenizer.encode(ctx_text_chars[:pos_chars], add_special_tokens=False)
    needle_start = len(prefix_ids)

    window = max_ctx_tokens - n_needle
    left_quota = int(window * needle_depth)
    start = max(0, needle_start - left_quota)
    end = min(len(ctx_ids), start + max_ctx_tokens)
    kept = ctx_ids[start:end]
    # verify survival in the kept window
    survived = tokenizer.decode(kept).find(needle) >= 0
    return tokenizer.decode(kept), survived


@torch.no_grad()
def evaluate_ruler_ppl(model, tokenizer, examples, device, max_ctx=2048, label="",
                       needle_depth=0.5, survival_log=None):
    """⑥ v2: needle-aware truncation + 100% survival assertion + full reference answer."""
    total_loss, total_tokens, n_used = 0.0, 0, 0

    for ex in tqdm(examples, desc=f"RULER {label}"):
        outputs = ex.get('outputs', [])
        if not outputs:
            continue
        answer = str(outputs[0])
        window_text, survived = needle_aware_truncate(ex['context'], answer, tokenizer,
                                                      max_ctx, needle_depth)
        if survival_log is not None:
            survival_log.append(survived)
        if not survived:
            continue  # counted in survival_log; final assertion decides

        query = ex['query']
        prompt = f"{window_text}\n\n{query}"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
        full_ids = torch.tensor([prompt_ids + answer_ids], device=device)
        p_len = len(prompt_ids)

        logits = model(full_ids, position_ids=torch.arange(full_ids.shape[1], device=device).unsqueeze(0)).logits
        tgt_logits = logits[0, p_len - 1:-1, :]
        tgt = full_ids[0, p_len:]
        if tgt.shape[0] == 0:
            raise RuntimeError(f"[{label}] empty answer span — scorer bug")
        loss = F.cross_entropy(tgt_logits, tgt, reduction='sum')
        total_loss += loss.item()
        total_tokens += tgt.shape[0]
        n_used += 1

    if n_used == 0:
        raise RuntimeError(f"[{label}] zero usable examples — check needle survival")
    avg_loss = total_loss / total_tokens
    return {'avg_loss': avg_loss, 'ppl': torch.exp(torch.tensor(avg_loss)).item(),
            'total_tokens': total_tokens, 'n_examples': n_used,
            'scorer_version': 'v2-needle-aware-full-answer'}


# thread cap: torch defaults to 128 OMP threads on this 256-core host; concurrent
# background jobs then oversubscribe CPU to >5000% (2026-09-07 ops incident)
torch.set_num_threads(8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/basemodel/llama3.2-1b")
    parser.add_argument("--ruler_path", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/dataset/ruler")
    parser.add_argument("--task", type=str, default="niah_single",
                        choices=["niah_single", "niah_multikey", "qa_squad", "qa_hotpotqa"])
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--patch_layers", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/project/freq-attention/results")
    parser.add_argument("--models", type=str, default="full,local")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 60)
    print(f"RULER EVAL (needle-aware 2,048-token eval window): {args.task}")
    print(f"Models: {args.models}, Samples: {args.max_samples}")
    print("=" * 60)

    examples = load_ruler_examples(args.ruler_path, args.task, args.max_samples)
    print(f"Loaded {len(examples)} examples")
    survival_log = []
    needle_depth = float(os.environ.get("RULER_NEEDLE_DEPTH", "0.5"))

    models_to_run = args.models.split(",")
    results = {}

    for model_name in models_to_run:
        print(f"\n--- {model_name} ---")
        gc.collect(); torch.cuda.empty_cache()

        if model_name == "full":
            model = load_llama_full(args.model_path, device=device)
            label = "full"
        elif model_name == "local":
            model = load_llama_local(args.model_path, window_size=args.window_size, device=device)
            label = f"local_ws{args.window_size}"
        elif model_name == "qcmsa":
            from src.eval.llama_loader import load_llama_qcmsa
            model = load_llama_qcmsa(args.model_path, window_size=args.window_size, device=device,
                                      patch_layers=args.patch_layers, ckpt_path=args.ckpt,
                                      num_experts=args.num_experts)
            label = f"qcmsa_ws{args.window_size}"
        else:
            continue

        r = evaluate_ruler_ppl(model, tokenizer, examples, device, label=label,
                                      needle_depth=needle_depth, survival_log=survival_log)
        # E-D: the "final assertion decides" comment was never backed by an assertion —
        # a silently-dropped needle sample would go unnoticed. Assert + record the rate.
        if survival_log:
            rate = sum(survival_log) / len(survival_log)
            assert rate == 1.0, (
                f"needle survival rate {rate:.3f} < 1.0 across {len(survival_log)} samples "
                f"(audit A.9)")
            r["needle_survival_rate"] = rate
            r["n_survival_checked"] = len(survival_log)
        results[model_name] = r
        print(f"  PPL: {r['ppl']:.1f} (tokens: {r['total_tokens']}, survival: "
              f"{r.get('needle_survival_rate', 'n/a')})")
        del model

    out = os.path.join(args.output_dir, f"ruler_{args.task}.json")
    json.dump(results, open(out, 'w'), indent=2)

    if "full" in results and "local" in results:
        gap = results["local"]["ppl"] - results["full"]["ppl"]
        print(f"\n  Local-Full gap: {gap:.0f} PPL")

    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
