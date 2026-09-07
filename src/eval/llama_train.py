"""
LLaMA QC-MSA Ablation Training — decisive long-context recovery experiment.

Adapts ablation_v2.py for LLaMA 3.2-1B. Key changes:
- Custom local mask (LLaMA has no HF sliding_window)
- RoPE-aware forward patching
- Eager attention backend

Usage:
  CUDA_VISIBLE_DEVICES=4 python -m src.eval.llama_train --exp_name llama_2k --train_len 2048 --steps 2000
  CUDA_VISIBLE_DEVICES=5 python -m src.eval.llama_train --exp_name llama_1k --train_len 1024 --steps 2000
"""

import torch, torch.nn.functional as F, gc, random, numpy as np, json, argparse, os, sys, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig
from src.models.llama_forwards import make_llama_qcmsa_forward as _make_llama_qcmsa_forward, make_llama_local_forward as _make_llama_local_forward
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import MethodType
from tqdm import tqdm


# ============================================================
# LLaMA patched forward functions: canonical versions live in src/models/llama_forwards.py
# (imported below as _make_llama_qcmsa_forward / _make_llama_local_forward)


def load_pg19_texts(pg19_path, max_examples=64):
    """Load PG-19 texts from parquet files (same format as perplexity.py)."""
    import pandas as pd
    data_dir = os.path.join(pg19_path, "data")
    parquet_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")])
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    dfs = [pd.read_parquet(pf) for pf in parquet_files]
    all_df = pd.concat(dfs, ignore_index=True)
    if max_examples is not None:
        all_df = all_df.head(max_examples)
    return all_df["text"].tolist()


def chunk_doc(tokens, L):
    return [tokens[i:i+L] for i in range(0, len(tokens) - L, L)]


@torch.no_grad()
def compute_ppl_llama(model, chunks, device, max_chunks=8):
    model.eval()
    tl, tt = 0.0, 0
    for c in chunks[:max_chunks]:
        inp = torch.tensor([c], device=device)
        pid = torch.arange(len(c), device=device).unsqueeze(0)
        out = model(inp, position_ids=pid)
        loss = F.cross_entropy(out.logits[0, :-1], torch.tensor(c[1:], device=device), reduction='sum')
        tl += loss.item()
        tt += len(c) - 1
    return torch.exp(torch.tensor(tl / tt)).item() if tt > 0 else float('nan')


# ============================================================
# Main
# ============================================================

# thread cap: torch defaults to 128 OMP threads on this 256-core host; concurrent
# background jobs then oversubscribe CPU to >5000% (2026-09-07 ops incident)
torch.set_num_threads(8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--model_path", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/basemodel/llama3.2-1b")
    parser.add_argument("--pg19_path", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/dataset/pg19")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--patch_layers", type=int, default=8)
    parser.add_argument("--no_phi_q", action="store_true",
                        help="Disable φ(q) in denominator (ablation)")
    parser.add_argument("--stationary", action="store_true",
                        help="Use stationary (query-independent) expert mixture")
    parser.add_argument("--train_len", type=str, default="1024")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_docs", type=int, default=32)
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhoujun/Auto-claude-code-research-in-sleep/project/freq-attention/results")
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--eval_manifest", type=str, default=None,
                        help="② fixed eval split JSON (make_eval_split.py); train split stays per-seed")
    parser.add_argument("--manifest_eval_chunks", type=int, default=16,
                        help="cap on chunks per eval length when using --eval_manifest")
    parser.add_argument("--train_gamma_only", action="store_true",
                        help="④ variant: freeze W_q/W_k/sigma at init, train ONLY gamma (is kernel learning necessary?)")
    args = parser.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_lengths = [int(x) for x in args.train_len.split(",")]
    eval_lengths = [min(256, l) for l in train_lengths] + train_lengths
    eval_lengths = sorted(set(eval_lengths))

    print("=" * 60)
    print(f"LLaMA QC-MSA Training: {args.exp_name}")
    print(f"  Window={args.window} M={args.experts} rank={args.rank} gate={'off' if args.stationary else 'on'} φ={'off' if args.no_phi_q else 'on'}")
    print(f"  Patch layers={args.patch_layers}")
    print(f"  Train lengths={train_lengths} Eval lengths={eval_lengths}")
    print(f"  Steps={args.steps} LR={args.lr}")
    print("=" * 60)

    # ---- Data ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_pg19_texts(args.pg19_path, max_examples=args.max_docs)
    doc_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=8192)
        if len(tokens) >= max(train_lengths):
            doc_tokens.append(tokens)
    random.shuffle(doc_tokens)
    n_train = int(len(doc_tokens) * 0.75)
    train_docs, eval_docs = doc_tokens[:n_train], doc_tokens[n_train:]
    print(f"Train docs: {len(train_docs)}, Eval docs: {len(eval_docs)}")

    eval_chunks = {}
    manifest_info = None
    if args.eval_manifest:
        # ② fixed eval split (drawn once, never re-drawn); train split stays per-seed.
        import pandas as pd
        man = json.load(open(args.eval_manifest))
        assert man["n_books"] == len(man["books"]), "manifest corrupted"
        man_books = []
        for f in sorted(set(b["file"] for b in man["books"])):
            df = pd.read_parquet(os.path.join(args.pg19_path, "data", f))
            for b in [x for x in man["books"] if x["file"] == f]:
                man_books.append(tokenizer.encode(df["text"].tolist()[b["row"]],
                                                  add_special_tokens=False, truncation=True, max_length=8192))
        for L in eval_lengths:
            chunks = []
            for toks in man_books:
                if len(toks) >= L + 1:
                    chunks.extend(chunk_doc(toks, L))
            eval_chunks[L] = chunks[:args.manifest_eval_chunks]
            print(f"  Eval@{L} [manifest]: {len(eval_chunks[L])} chunks")
        manifest_info = {"path": args.eval_manifest, "sha256": man["sha256"],
                         "n_chunks_per_len": {str(L): len(eval_chunks[L]) for L in eval_lengths}}
    else:
        for L in eval_lengths:
            chunks = []
            for doc in eval_docs:
                chunks.extend(chunk_doc(doc, L))
            eval_chunks[L] = chunks[:8]
            print(f"  Eval@{L}: {len(eval_chunks[L])} chunks")

    # ---- Baselines ----
    print("\n--- Baselines ---")
    baseline = {}
    # Local baseline
    gc.collect(); torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16,
                                              attn_implementation="eager").to(device).eval()
    for layer in m.model.layers:
        layer.self_attn.forward = MethodType(_make_llama_local_forward(args.window), layer.self_attn)
    for L in eval_lengths:
        baseline[f"local_{L}"] = compute_ppl_llama(m, eval_chunks[L], device)
        print(f"  Local(ws={args.window}) @{L}: {baseline[f'local_{L}']:.1f}")
    del m

    # Full baseline (only for shorter lengths to avoid OOM)
    gc.collect(); torch.cuda.empty_cache()
    m_full = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to(device).eval()
    for L in eval_lengths:
        if L <= 2048:
            baseline[f"full_{L}"] = compute_ppl_llama(m_full, eval_chunks[L], device)
            print(f"  Full @{L}: {baseline[f'full_{L}']:.1f}")
    del m_full

    # ---- Load model for QC-MSA ----
    gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16,
                                                  attn_implementation="eager").to(device).eval()

    kcfg = QCMSAConfig(
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        feature_rank=args.rank,
        num_experts=args.experts,
        kernel_types=tuple(["exponential"] * args.experts),
        use_gate=not args.stationary,
        use_phi_q=not args.no_phi_q,
        window_size=args.window,
        dtype=torch.float32,
    )

    n_layer = len(model.model.layers)
    patch_n = min(args.patch_layers, n_layer)
    start_layer = n_layer - patch_n

    if args.experts >= 4:
        multiscale_sigmas = [3.0, 4.0, 5.0, 6.0][:args.experts]
    else:
        multiscale_sigmas = list(np.linspace(3.0, 6.0, args.experts))

    for i, layer in enumerate(model.model.layers):
        if i >= start_layer:
            sp = CorrectSpectralAttention(kcfg).to(device=device, dtype=torch.float32)
            sp.log_sigma.data = torch.tensor(multiscale_sigmas, device=device)
            layer.self_attn.qcmsa_spectral = sp
            layer.self_attn.forward = MethodType(_make_llama_qcmsa_forward(sp, args.window), layer.self_attn)
        else:
            layer.self_attn.forward = MethodType(_make_llama_local_forward(args.window), layer.self_attn)
            layer.self_attn.qcmsa_spectral = None

    print(f"Patched layers {start_layer}-{n_layer-1} ({patch_n}/{n_layer})")
    print(f"Sigmas: exp({multiscale_sigmas}) = {[round(np.exp(s),1) for s in multiscale_sigmas]}")

    # ---- Freeze all except spectral ----
    for n, p in model.named_parameters():
        p.requires_grad = False
    spectral_params = []
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'qcmsa_spectral') and layer.self_attn.qcmsa_spectral is not None:
            sp_mod = layer.self_attn.qcmsa_spectral
            if args.train_gamma_only:
                for p in sp_mod.parameters():
                    p.requires_grad = False
                sp_mod.gamma.requires_grad = True
                spectral_params.append(sp_mod.gamma)
            else:
                for p in sp_mod.parameters():
                    p.requires_grad = True
                    spectral_params.append(p)

    print(f"Trainable: spectral={sum(p.numel() for p in spectral_params):,}")

    # ---- Pre-training eval ----
    pre_ppl = {}
    for L in eval_lengths:
        pre_ppl[L] = compute_ppl_llama(model, eval_chunks[L], device)
        print(f"Pre-train @{L}: {pre_ppl[L]:.1f}")

    # ---- Training ----
    opt = torch.optim.AdamW(spectral_params, lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    best = {L: float('inf') for L in eval_lengths}
    history = []
    guard_skips = 0  # F3: total_norm skip counter (NaN-grad steps)

    pbar = tqdm(range(args.steps), desc=f"[{args.exp_name}]")
    for step in pbar:
        model.train()
        doc = random.choice(train_docs)
        train_L = random.choice(train_lengths)
        chunks = chunk_doc(doc, train_L)
        if not chunks:
            continue
        c = random.choice(chunks)
        inp = torch.tensor([c], device=device)
        pid = torch.arange(len(c), device=device).unsqueeze(0)

        out = model(input_ids=inp, position_ids=pid)
        loss = F.cross_entropy(out.logits[0, :-1], torch.tensor(c[1:], device=device), reduction='mean')

        if torch.isnan(loss):
            print(f"\nNaN at step {step}, skipping")
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(spectral_params, 0.5)
        if not torch.isfinite(total_norm):
            # ①-R: finite loss can still produce NaN grads (fp16 base overflow in backward);
            # clip_grad_norm_ then propagates NaN into every param (M4L4_s3 / M4L8 retry-4 root cause)
            guard_skips += 1  # F3: instrumented per expert review (was silent continue)
            opt.zero_grad(); continue
        opt.step()
        sch.step()

        # Clamp gamma
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'qcmsa_spectral') and layer.self_attn.qcmsa_spectral is not None:
                layer.self_attn.qcmsa_spectral.gamma.data.clamp_(-0.3, 0.3)

        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            eval_strs = []
            eval_vals = {}  # A7: single forward pass per length; reused by history (was 2x redundant)
            for L in eval_lengths:
                p = compute_ppl_llama(model, eval_chunks[L], device)
                eval_vals[L] = p
                best[L] = min(best[L], p)
                eval_strs.append(f"@{L}={p:.1f}")

            gammas = []
            sigmas = []
            for layer in model.model.layers:
                if hasattr(layer.self_attn, 'qcmsa_spectral') and layer.self_attn.qcmsa_spectral is not None:
                    gammas.append(layer.self_attn.qcmsa_spectral.gamma.item())
                    sigmas.append([round(v, 5) for v in layer.self_attn.qcmsa_spectral.log_sigma.detach().tolist()])
            g_mean = np.mean(gammas) if gammas else 0
            # ①-R σ trajectory (per-step, per-layer, per-expert): the decisive evidence for
            # "does the model want longer reach" -> 99%-mass reach(t) curve (plan E9/E9b).
            sigma_mean = [round(v, 5) for v in np.mean(sigmas, axis=0)] if sigmas else []

            pbar.set_postfix_str(f"loss={loss.item():.3f} {' '.join(eval_strs)} g={g_mean:.4f} σ={sigma_mean}")
            history.append({'step': step, 'loss': loss.item(), 'gamma_mean': g_mean,
                            'log_sigma_mean': sigma_mean,
                           **{f'ppl_{L}': eval_vals[L] for L in eval_lengths}})  # A7: reuse

    # ---- Final ----
    model.eval()
    final_ppl = {}
    for L in eval_lengths:
        final_ppl[L] = compute_ppl_llama(model, eval_chunks[L], device)

    # ---- Save checkpoint ----
    ckpt_file = os.path.join(args.output_dir, f"{args.exp_name}.pt")
    spectral_state = {}
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, 'qcmsa_spectral') and layer.self_attn.qcmsa_spectral is not None:
            spectral_state[str(i)] = layer.self_attn.qcmsa_spectral.state_dict()
    torch.save(spectral_state, ckpt_file)

    results = {
        'exp_name': args.exp_name,
        'config': {'window': args.window, 'experts': args.experts, 'rank': args.rank,
                   'patch_layers': patch_n, 'steps': args.steps, 'lr': args.lr,
                   'train_lengths': train_lengths, 'multiscale_sigmas': multiscale_sigmas},
        'baseline': baseline, 'pre_train': pre_ppl, 'final': final_ppl,
        'best': {str(L): best[L] for L in eval_lengths},
        'history': history,
        'n_spectral': sum(p.numel() for p in spectral_params),
        'provenance': {
            'guard_skips': guard_skips,
            'eval_manifest': manifest_info,
            'hard_zero_empty_far': True,
            'code_sha256': {name: hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
                            for name, path in [
                                ('kernel', os.path.join(os.path.dirname(__file__), '..', 'kernels', 'correct_spectral_attention.py')),
                                ('forwards', os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'models', 'llama_forwards.py')),
                                ('train', os.path.abspath(__file__))]},
        'cache_fix': 'train: no kernel-FFT caching under grad; inference: key=sigma (0a-4)',
        },
    }

    result_file = os.path.join(args.output_dir, f"{args.exp_name}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== RESULTS: {args.exp_name} ===")
    for L in eval_lengths:
        local = baseline.get(f'local_{L}', 0)
        full = baseline.get(f'full_{L}', 0)
        delta_local = local - final_ppl[L]
        pct_local = delta_local / local * 100 if local > 0 else 0
        delta_full = final_ppl[L] - full
        print(f"  @{L}: pre={pre_ppl[L]:.1f} → final={final_ppl[L]:.1f} (best={best[L]:.1f})")
        print(f"       vs local={local:.1f}: Δ={delta_local:+.1f} ({pct_local:+.1f}%)")
        if full > 0:
            gap = local - full
            closed = delta_local / gap * 100 if gap > 0 else 0
            print(f"       vs full={full:.1f}: gap={gap:.0f}, closed={closed:.1f}%")

    print(f"\nSaved: {result_file}, {ckpt_file}")


if __name__ == "__main__":
    main()
