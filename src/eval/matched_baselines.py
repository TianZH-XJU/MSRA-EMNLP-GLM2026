"""
⑦ Matched-budget baselines (PHASE1_EXPERIMENT_PLAN step 7) — "Is the spectral form necessary?"

Two parameter-budget-matched corrections on the SAME local-attention base, SAME data protocol,
SAME fixed eval manifest, SAME steps. OPTIMIZER CALIBER DIFFERS (disclosed, A2): lr 1e-4 and
grad-clip 1.0 here vs MSRA's lr 2e-4-era protocol and clip 0.5 — the adapter-standard lr is MORE
conservative, so the adapters' dominance (78-88% vs MSRA 12.5%) is, if anything, understated:
  --baseline lora    rank-1 LoRA on o_proj of the last L layers  (2*d_model per layer)
  --baseline dwconv  causal depthwise conv (kernel 8) on o_proj input (d_model*8 per layer)

Comparison arms (results/fixed_split_eval.json): MSRA M=1 L=8 = 12.3%, M=4 L=4 = 8.6%,
gamma-only = 9.5%, random-untrained = 0%.

Usage:
  CUDA_VISIBLE_DEVICES=5 python -m src.eval.matched_baselines --baseline lora \
      --exp_name R_bl_lora_s2 --seed 2 --eval_manifest results/eval_split_256.json
"""

import argparse, gc, hashlib, json, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.llama_forwards import make_llama_local_forward
from src.eval.llama_train import load_pg19_texts, chunk_doc, compute_ppl_llama


class LoRAR1OWrapper(nn.Module):
    """y = o_proj(x) + B(A(x)); rank-1, 2*d_model trainable params per layer."""
    def __init__(self, o_proj, d):
        super().__init__()
        self.o_proj = o_proj
        self.lora_A = nn.Linear(d, 1, bias=False)
        self.lora_B = nn.Linear(1, d, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.o_proj(x) + self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))).to(x.dtype)


class CausalDWConvWrapper(nn.Module):
    """y = o_proj(x) + causal depthwise conv(x); d_model*kernel params per layer."""
    def __init__(self, o_proj, d, kernel=8):
        super().__init__()
        self.o_proj = o_proj
        self.conv = nn.Conv1d(d, d, kernel, groups=d, bias=False)
        nn.init.zeros_(self.conv.weight)
        self.kernel = kernel

    def forward(self, x):
        xc = x.transpose(1, 2).to(self.conv.weight.dtype)
        pad = F.pad(xc, (self.kernel - 1, 0))  # causal: pad left only
        return self.o_proj(x) + self.conv(pad).transpose(1, 2).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["lora", "dwconv"], required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--model_path", default="/data1/zhoujun/Auto-claude-code-research-in-sleep/basemodel/llama3.2-1b")
    ap.add_argument("--pg19_path", default="/data1/zhoujun/Auto-claude-code-research-in-sleep/dataset/pg19")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--patch_layers", type=int, default=8)
    ap.add_argument("--train_len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--max_docs", type=int, default=64)
    ap.add_argument("--eval_manifest", default="results/eval_split_256.json")
    ap.add_argument("--manifest_eval_chunks", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # data: training pool per old protocol; eval from the fixed manifest (same as ①-R)
    texts = load_pg19_texts(args.pg19_path, max_examples=args.max_docs)
    doc_tokens = [tok.encode(t, add_special_tokens=False, truncation=True, max_length=8192)
                  for t in texts]
    doc_tokens = [d for d in doc_tokens if len(d) >= args.train_len + 1]
    random.shuffle(doc_tokens)
    n_train = int(len(doc_tokens) * 0.75)
    train_docs = doc_tokens[:n_train]
    train_chunks = []
    for doc in train_docs:
        train_chunks.extend(chunk_doc(doc, args.train_len))
    print(f"train chunks: {len(train_chunks)}")

    import pandas as pd
    man = json.load(open(args.eval_manifest))
    man_books = []
    for f in sorted(set(b["file"] for b in man["books"])):
        df = pd.read_parquet(os.path.join(args.pg19_path, "data", f))
        for b in [x for x in man["books"] if x["file"] == f]:
            man_books.append(tok.encode(df["text"].tolist()[b["row"]],
                                        add_special_tokens=False, truncation=True, max_length=8192))
    eval_chunks = []
    for toks in man_books:
        if len(toks) >= args.train_len + 1:
            eval_chunks.extend(chunk_doc(toks, args.train_len))
    eval_chunks = eval_chunks[:args.manifest_eval_chunks]
    print(f"eval chunks [manifest {man['sha256'][:8]}]: {len(eval_chunks)}")

    # model: ALL layers local attention; last L get the baseline correction on o_proj
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16,
                                                 attn_implementation="eager").to(device).eval()
    d_model = model.config.hidden_size
    n_layer = len(model.model.layers)
    start = n_layer - min(args.patch_layers, n_layer)
    trainables = []
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.forward = MethodType(make_llama_local_forward(args.window), layer.self_attn)
        if li >= start:
            o = layer.self_attn.o_proj
            if args.baseline == "lora":
                # NOTE: .to(dtype=fp32) would also convert the wrapped fp16 o_proj —
                # cast only the trainable adapters, keep o_proj in the model's dtype
                wrap = LoRAR1OWrapper(o, d_model).to(device=device)
                wrap.lora_A = wrap.lora_A.to(dtype=torch.float32)
                wrap.lora_B = wrap.lora_B.to(dtype=torch.float32)
                trainables += [wrap.lora_A.weight, wrap.lora_B.weight]
            else:
                wrap = CausalDWConvWrapper(o, d_model, kernel=8).to(device=device)
                wrap.conv = wrap.conv.to(dtype=torch.float32)
                trainables += [wrap.conv.weight]
            layer.self_attn.o_proj = wrap
    trainable_ids = {id(p) for p in trainables}
    for n, p in model.named_parameters():
        p.requires_grad = (id(p) in trainable_ids)  # A9: id-hash membership; valid because params are not cloned
    n_par = sum(p.numel() for p in trainables)
    print(f"baseline={args.baseline} trainables={n_par:,} (MSRA M1L8 = 18,448)")

    # baselines
    baseline = {}
    m_loc = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16,
                                                 attn_implementation="eager").to(device).eval()
    for layer in m_loc.model.layers:
        layer.self_attn.forward = MethodType(make_llama_local_forward(args.window), layer.self_attn)
    baseline["local_512"] = compute_ppl_llama(m_loc, eval_chunks, device)
    del m_loc; gc.collect(); torch.cuda.empty_cache()
    m_full = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to(device).eval()
    baseline["full_512"] = compute_ppl_llama(m_full, eval_chunks, device)
    del m_full; gc.collect(); torch.cuda.empty_cache()
    print(f"baselines: {baseline}")

    opt = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    history, best = [], {"512": float("inf")}
    rng = np.random.RandomState(args.seed)
    model.train()
    pbar = tqdm(range(args.steps))
    for step in pbar:
        c = train_chunks[rng.randint(len(train_chunks))]
        inp = torch.tensor([c], device=device)
        loss = model(inp, labels=inp).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
        opt.step(); sch.step(); opt.zero_grad()
        for layer in model.model.layers:
            o = layer.self_attn.o_proj
            if isinstance(o, LoRAR1OWrapper):
                o.lora_B.weight.data.clamp_(-1.0, 1.0)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            p = compute_ppl_llama(model, eval_chunks, device)
            best["512"] = min(best["512"], p)
            history.append({"step": step, "loss": loss.item(), "ppl_512": p})
            pbar.set_postfix_str(f"loss={loss.item():.3f} @512={p:.1f}")
            model.train()
    model.eval()
    final_ppl = compute_ppl_llama(model, eval_chunks, device)

    ckpt = {str(i): {"o_proj_wrap": {k: v.detach().cpu() for k, v in layer.self_attn.o_proj.state_dict().items()
                                     if any(o is layer.self_attn.o_proj for o in [layer.self_attn.o_proj])}}
            for i, layer in enumerate(model.model.layers) if i >= start}
    torch.save(ckpt, os.path.join(args.output_dir, f"{args.exp_name}.pt"))
    results = {"exp_name": args.exp_name, "baseline": args.baseline,
               "config": {"patch_layers": args.patch_layers, "steps": args.steps, "lr": args.lr,
                          "seed": args.seed, "n_trainables": n_par},
               "baseline": baseline, "final": {"512": final_ppl},
               "best": {str(L): best[L] for L in best}, "history": history,
               "provenance": {"manifest": args.eval_manifest, "manifest_sha256": man["sha256"],
                              "code_sha256": hashlib.sha256(open(os.path.join(os.path.dirname(__file__),
                                  '..', 'kernels', 'correct_spectral_attention.py'), 'rb').read()).hexdigest()[:16]}}
    json.dump(results, open(os.path.join(args.output_dir, f"{args.exp_name}.json"), 'w'), indent=2)
    loc = baseline["local_512"]; ful = baseline["full_512"]
    print(f"\n=== {args.exp_name} ===")
    print(f"  final@512={final_ppl:.1f} best={best['512']:.1f} local={loc:.1f} full={ful:.1f}")
    print(f"  gap closed = {(loc-final_ppl)/(loc-ful)*100:.1f}%  (gain {(loc-final_ppl)/loc*100:+.1f}%)")


if __name__ == "__main__":
    main()
