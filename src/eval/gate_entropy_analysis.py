"""E-E: gate-artifact anchor for the §5.2(3) selectivity claim.

Re-derives the two numbers quoted in the paper from the POST-fix gate-on checkpoint,
with provenance (the pre-fix query_gate_analysis.json dates from the cache-bug era
and cannot back any current claim; the 69%/9x figures had no surviving artifact).

Protocol (documented here because the paper cites it):
  - entropy ratio  = mean H(lambda) / ln(M) over all patched layers, positions and
    fixed-split chunks. 1.0 = uniform (decorative), <1 = selective.
  - perturbation response = mean |lambda(q) - lambda(q')| at the last position when
    the QUERY token is replaced by a random vocabulary token, divided by the same
    quantity when a FAR-CONTEXT token (position 64) is replaced instead. The gate
    is query-conditioned by design, so a ratio >> 1 supports "genuinely selective".

Usage:
  CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 python -m src.eval.gate_entropy_analysis \
      --ckpt results/R_M4L8_s2b.pt --out results/gate_analysis_v2.json
"""
import argparse, gc, json, os, sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig
from src.models.llama_forwards import patch_llama_layers
from src.eval.eval_fixed_split import MP, load_chunks

torch.set_num_threads(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/R_M4L8_s2b.pt")
    ap.add_argument("--manifest", default="results/eval_split_256.json")
    ap.add_argument("--n_chunks", type=int, default=8)
    ap.add_argument("--n_probe_chunks", type=int, default=4)
    ap.add_argument("--n_perturb", type=int, default=8)
    ap.add_argument("--out", default="results/gate_analysis_v2.json")
    args = ap.parse_args()
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(MP)
    model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16,
                                                 attn_implementation="eager").to(device).eval()
    ckd = torch.load(args.ckpt, map_location=device)
    M = len(next(iter(ckd.values()))["log_sigma"])
    assert "gate.weight" in next(iter(ckd.values())), "gate-off ckpt — nothing to analyse"

    def factory():
        return CorrectSpectralAttention(QCMSAConfig(head_dim=64, feature_rank=16, num_experts=M,
            kernel_types=("exponential",) * M, use_gate=True, use_phi_q=True,
            window_size=128, dtype=torch.float32))
    patch_llama_layers(model, 128, patch_last_n=len(ckd), spectral_factory=factory, ckpt=ckd, device=device)

    # capture lambda = softmax(gate(q_normed)) per patched layer
    lam_capture = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        sp = getattr(layer.self_attn, 'qcmsa_spectral', None)
        if sp is None or not hasattr(sp, 'gate'):
            continue
        def mk(store):
            def hook(mod, inp, out):
                store.append(F.softmax(out.detach().float(), dim=-1))
            return hook
        hooks.append(sp.gate.register_forward_hook(mk(lam_capture.setdefault(li, []))))

    chunks, _ = load_chunks(tok, args.manifest, 512, args.n_chunks)
    vocab = model.config.vocab_size
    rng = torch.Generator().manual_seed(42)

    with torch.no_grad():
        for c in chunks:
            _ = model(torch.tensor([c], device=device))
    ent_nats, per_layer = {}, {}
    for li, caps in lam_capture.items():
        lam = torch.cat(caps, dim=0)              # (B*H*T_total, M)
        ent = -(lam * lam.clamp_min(1e-12).log()).sum(-1)
        ent_nats[li] = float(ent.mean())
        uni = torch.full_like(lam, 1.0 / M)
        kl = (lam * (lam.clamp_min(1e-12).log() - float(torch.log(torch.tensor(1.0 / M))))).sum(-1)
        per_layer[li] = {"mean_entropy_nats": float(ent.mean()),
                         "entropy_ratio": float(ent.mean() / torch.log(torch.tensor(float(M)))),
                         "mean_kl_vs_uniform": float(kl.mean())}
    for h in hooks: h.remove()
    for li in lam_capture: lam_capture[li].clear()
    entropy_ratio_mean = float(sum(v["entropy_ratio"] for v in per_layer.values()) / len(per_layer))

    # perturbation probes on fresh hooks
    probe_chunks, _ = load_chunks(tok, args.manifest, 512, args.n_probe_chunks + args.n_chunks)
    probe_chunks = probe_chunks[args.n_chunks:args.n_chunks + args.n_probe_chunks]
    lam_probe = {}
    hooks2 = []
    for li, layer in enumerate(model.model.layers):
        sp = getattr(layer.self_attn, 'qcmsa_spectral', None)
        if sp is None or not hasattr(sp, 'gate'):
            continue
        def mk(store):
            def hook(mod, inp, out):
                store.append(F.softmax(out.detach().float(), dim=-1))
            return hook
        hooks2.append(sp.gate.register_forward_hook(mk(lam_probe.setdefault(li, []))))

    def last_lam(input_ids):
        for li in lam_probe: lam_probe[li].clear()
        with torch.no_grad():
            _ = model(input_ids)
        return {li: v[-1][-1, -1, :] for li, v in lam_probe.items()}  # last layer-step, last pos

    d_query, d_context = [], []
    with torch.no_grad():
        for c in probe_chunks:
            base = last_lam(torch.tensor([c], device=device))
            for _ in range(args.n_perturb):
                rt = int(torch.randint(100, vocab - 1, (1,), generator=rng))
                cq = c.copy(); cq[-1] = rt          # query-token perturbation
                cf = c.copy(); cf[64] = rt          # far-context perturbation (same token)
                lq = last_lam(torch.tensor([cq], device=device))
                lf = last_lam(torch.tensor([cf], device=device))
                for li in base:
                    d_query.append(float((base[li] - lq[li]).abs().mean()))
                    d_context.append(float((base[li] - lf[li]).abs().mean()))
    for h in hooks2: h.remove()
    import statistics
    dq, dc = statistics.mean(d_query), statistics.mean(d_context)
    ratio = dq / dc if dc > 0 else float("nan")

    out = {
        "provenance": {"ckpt": args.ckpt, "protocol": __doc__.split("Protocol (")[1].split("Usage:")[0].strip(),
                       "n_chunks": args.n_chunks, "M": M,
                       "code": "src/eval/gate_entropy_analysis.py"},
        "entropy_ratio_mean": entropy_ratio_mean,
        "per_layer": per_layer,
        "perturbation": {"query_token_mean_dlambda": dq, "far_context_token_mean_dlambda": dc,
                         "query_over_context_ratio": ratio,
                         "n_probes": len(d_query)},
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"entropy_ratio_mean = {entropy_ratio_mean:.4f}  (paper claim: selectivity)")
    print(f"perturbation ratio (query/context) = {ratio:.2f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
