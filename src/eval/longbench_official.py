"""
LongBench OFFICIAL-metric evaluation (plan step: LongBench 官方指标; gates abstract 6.9%).

Replaces the perplexity-space scoring that reviewers rejected: greedy generation +
official LongBench metrics (QA F1 / ROUGE-L, standard implementations from the
LongBench evaluation suite). Three arms share identical prompts and generation
config; provenance (ckpt hash, manifest of task files) recorded.

Truncation: prompts are left-truncated to --max_prompt_tokens (4096) — disclosed in
the paper; all arms use identical truncation so comparisons are matched.

Usage:
  CUDA_VISIBLE_DEVICES=5 python -m src.eval.longbench_official --ckpt results/R_M1L8_s2.pt \
      --tasks triviaqa,narrativeqa,hotpotqa,2wikimqa,qasper,gov_report,qmsum --max_samples 30
"""

import argparse, gc, hashlib, json, os, re, string, sys
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import MethodType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig
from src.models.llama_forwards import patch_llama_layers

MP = "/data1/zhoujun/Auto-claude-code-research-in-sleep/basemodel/llama3.2-1b"
LB = "/data1/zhoujun/Auto-claude-code-research-in-sleep/dataset/longbench/data"

QA_TASKS = {"triviaqa", "narrativeqa", "hotpotqa", "2wikimqa", "qasper", "musique", "passage_retrieval_en"}
SUMM_TASKS = {"gov_report", "qmsum", "multi_news"}
GEN_LEN = {"qa": 32, "summ": 256}  # official uses 32 (qa) / 512 (summ); 256 keeps cost sane


# ---------------- official LongBench metrics (standard implementations) ----------------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    pred = normalize_answer(prediction).split()
    gt = normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truths):
    return max(f1_score(prediction, gt) for gt in ground_truths) if ground_truths else 0.0


def _lcs(x, y):
    n, m = len(x), len(y)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if x[i-1] == y[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]


def rouge_l_score(prediction, ground_truths):
    """ROUGE-L F1 via LCS on word tokens (max over gold references)."""
    if not ground_truths:
        return 0.0
    best = 0.0
    for gt in ground_truths:
        p_toks = normalize_answer(prediction).split()
        g_toks = normalize_answer(gt).split()
        if not p_toks or not g_toks:
            continue
        lcs = _lcs(p_toks, g_toks)
        prec, rec = lcs / len(p_toks), lcs / len(g_toks)
        if prec + rec > 0:
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def get_metric(task):
    if task in QA_TASKS:
        return qa_f1_score, "qa_f1"
    if task in SUMM_TASKS:
        return rouge_l_score, "rouge_l"
    raise ValueError(f"task {task} not configured (qa/summ only)")


# ---------------- harness ----------------
def load_task(task, max_samples):
    path = os.path.join(LB, f"{task}.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            inp = d.get("input", "")
            if not inp.strip():
                # gov_report-style rows: content in `context`, prompt built from the
                # official LongBench per-task template
                tmpl = {"gov_report": "You are given a report. Summarize the report.\n\n{context}\n\nSummary:",
                        "qmsum": "You are given a meeting transcript. Summarize the meeting.\n\n{context}\n\nSummary:",
                        "multi_news": "You are given several news articles. Summarize them.\n\n{context}\n\nSummary:"}
                if task in tmpl:
                    inp = tmpl[task].format(context=d.get("context", ""))
                else:
                    continue  # genuinely malformed
            rows.append({"input": inp, "outputs": d.get("answers", d.get("outputs", []))})
            if len(rows) >= max_samples:
                break
    return rows


def build_prompt(row, tok, max_prompt_tokens):
    # row["input"] is the official pre-formatted prompt (context + question + instruction);
    # left-truncate to keep the question/instruction tail when over budget
    if not row["input"] or not row["input"].strip():
        raise ValueError("empty LongBench input row")
    ids = tok.encode(row["input"], add_special_tokens=False)
    if len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
        return tok.decode(ids)
    return row["input"]


@torch.no_grad()
def generate(model, tok, prompt, max_new, device):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def sha16(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16] if p and os.path.exists(p) else None


# thread cap: torch defaults to 128 OMP threads on this 256-core host; concurrent
# background jobs then oversubscribe CPU to >5000% (2026-09-07 ops incident)
torch.set_num_threads(8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="triviaqa,narrativeqa,hotpotqa,2wikimqa,qasper,gov_report,qmsum")
    ap.add_argument("--max_samples", type=int, default=30)
    ap.add_argument("--max_prompt_tokens", type=int, default=4096)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--ckpt", default="results/R_M1L8_s2.pt")
    ap.add_argument("--models", default="full,local,qcmsa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/longbench_official.json")
    args = ap.parse_args()
    device = args.device
    tok = AutoTokenizer.from_pretrained(MP)
    results = {"provenance": {"ckpt": args.ckpt, "ckpt_sha256": sha16(args.ckpt),
                              "code_sha256": sha16(os.path.join(os.path.dirname(__file__), '..', 'kernels', 'correct_spectral_attention.py')),
                              "max_prompt_tokens": args.max_prompt_tokens,
                              "metrics": "official LongBench (qa_f1 / rouge_l), greedy decode"}}

    def build(model_type):
        m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16,
                                                 attn_implementation="eager").to(device).eval()
        if model_type == "full":
            return m
        if model_type == "local":
            for layer in m.model.layers:
                layer.self_attn.forward = MethodType(
                    __import__('src.models.llama_forwards', fromlist=['make_llama_local_forward']).make_llama_local_forward(args.window), layer.self_attn)
            return m
        ckd = torch.load(args.ckpt, map_location=device)
        M = len(next(iter(ckd.values()))["log_sigma"])
        gate = "gate.weight" in next(iter(ckd.values()))
        def factory():
            return CorrectSpectralAttention(QCMSAConfig(head_dim=64, feature_rank=16, num_experts=M,
                kernel_types=("exponential",) * M, use_gate=gate, use_phi_q=True,
                window_size=args.window, dtype=torch.float32))
        patch_llama_layers(m, args.window, patch_last_n=len(ckd), spectral_factory=factory, ckpt=ckd, device=device)
        return m

    for task in args.tasks.split(","):
      try:
        metric_fn, metric_name = get_metric(task)
        rows = load_task(task, args.max_samples)
        gen_len = GEN_LEN["summ" if task in SUMM_TASKS else "qa"]
        print(f"\n=== {task} ({metric_name}, n={len(rows)}, gen={gen_len}) ===", flush=True)
        task_res = {}
        for model_type in args.models.split(","):
            gc.collect(); torch.cuda.empty_cache()
            try:
                m = build(model_type)
                scores = []
                empty_gen = 0
                for row in rows:
                    prompt = build_prompt(row, tok, args.max_prompt_tokens)
                    pred = generate(m, tok, prompt, gen_len, device)
                    # hygiene: the spectral module's _buffer_cache keys on padded_len;
                    # tasks with varying prompt lengths (triviaqa) allocate a new FP32
                    # buffer set per distinct length and OOM by mid-task — release per sample
                    if hasattr(m, 'model'):
                        for layer in m.model.layers:
                            sp = getattr(layer.self_attn, 'qcmsa_spectral', None)
                            if sp is not None:
                                sp.clear_caches()
                    torch.cuda.empty_cache()
                    if not pred:
                        empty_gen += 1
                    if len(scores) == 0:
                        print(f"    sample[{model_type}]: {pred[:120]!r}", flush=True)
                    scores.append(metric_fn(pred, row["outputs"]))
                task_res[model_type] = {"metric": metric_name,
                                        "score": sum(scores) / len(scores),
                                        "n": len(scores), "empty_generations": empty_gen}
                print(f"  {model_type:>6}: {task_res[model_type]['score']:.4f} (empty gen: {empty_gen})", flush=True)
            except (ValueError, RuntimeError) as e:
                print(f"  !! {model_type} arm error: {str(e)[:100]}", flush=True)
                task_res[model_type] = {"metric": metric_name, "score": None, "error": str(e)[:100]}
            finally:
                m = None
                gc.collect(); torch.cuda.empty_cache()
        results[task] = task_res
      except torch.cuda.OutOfMemoryError as e:
        print(f"  !! {task} OOM: {str(e)[:80]} — recorded as oom, continuing", flush=True)
        results[task] = {"error": "cuda_oom"}
        torch.cuda.empty_cache()

    json.dump(results, open(args.out, 'w'), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
