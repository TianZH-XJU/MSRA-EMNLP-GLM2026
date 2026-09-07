"""V8-methodology fp32 discrimination for fp16-bit-identical checkpoint pairs.

Recomputes the canonical fixed-split 16-chunk eval @512 under a full-fp32
model (base fp32 + spectral fp32). Purpose: when two seeds' fp16 canonical
evals come out bit-identical (quantization-floor suspicion), fp32 evaluation
must discriminate them — distinct fp32 values confirm the seeds genuinely
differ and the fp16 identity is the quantization floor, not a harness bug
(same methodology as v8_discrimination.json for R_M1L8 s2/s3/s4).

Paper slots: Table 5 seed columns; audit appendix A.5 / fp32-stability note.

Usage:
  CUDA_VISIBLE_DEVICES=5 python -m src.eval.fp32_discriminate \
      --ckpts results/R_M4L8gateoff_s2.pt results/R_M4L8gateoff_s3.pt \
      --out results/v8_gateoff_fp32.json
"""
import argparse, gc, json, os
from types import MethodType

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig
from src.models.llama_forwards import patch_llama_layers
from src.eval.eval_fixed_split import MP, load_chunks, ppl, sha16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/eval_split_256.json")
    ap.add_argument("--n_chunks", type=int, default=16)
    ap.add_argument("--lengths", default="512")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device
    lengths = [int(x) for x in args.lengths.split(",")]

    tok = AutoTokenizer.from_pretrained(MP)
    chunk_sets = {}
    for L in lengths:
        chunks, _ = load_chunks(tok, args.manifest, L, args.n_chunks)
        assert len(chunks) >= 8, f"manifest yielded only {len(chunks)} chunks @{L}"
        chunk_sets[L] = chunks
    print(f"fixed chunks: { {L: len(c) for L, c in chunk_sets.items()} }", flush=True)

    out = {"provenance": {"manifest": args.manifest,
                          "manifest_sha256": json.load(open(args.manifest))["sha256"],
                          "code_sha256": sha16(os.path.join(os.path.dirname(__file__), '..', 'kernels', 'correct_spectral_attention.py')),
                          "n_chunks": args.n_chunks, "dtype": "float32"},
           "arms": {}}
    for ck in args.ckpts:
        name = os.path.splitext(os.path.basename(ck))[0]
        m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float32, attn_implementation="eager").to(device).eval()
        ckd = torch.load(ck, map_location=device)
        n_layers_patched = len(ckd)
        M = len(next(iter(ckd.values()))["log_sigma"])
        gate = "gate.weight" in next(iter(ckd.values()))
        def factory():
            return CorrectSpectralAttention(QCMSAConfig(
                head_dim=64, feature_rank=16, num_experts=M, kernel_types=("exponential",) * M,
                use_gate=gate, use_phi_q=True, window_size=128, dtype=torch.float32))
        patch_llama_layers(m, 128, patch_last_n=n_layers_patched, spectral_factory=factory, ckpt=ckd, device=device)
        # load-fidelity assertion (A4)
        for li, layer in enumerate(m.model.layers):
            sp = getattr(layer.self_attn, 'qcmsa_spectral', None)
            if sp is None or str(li) not in ckd: continue
            for pk, pv in ckd[str(li)].items():
                named = dict(sp.named_parameters())
                if pk in named and not torch.allclose(named[pk].detach().float().cpu(), pv.float().cpu(), atol=1e-6):
                    raise RuntimeError(f"load fidelity violated: {name} layer {li} {pk}")
        out["arms"][name] = {str(L): ppl(m, chunk_sets[L], device) for L in lengths}
        out["arms"][name + "_meta"] = {"ckpt_sha256": sha16(ck), "n_patched_layers": n_layers_patched}
        print(f"{name} (fp32): {out['arms'][name]}", flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()

    vals = [v["512"] for k, v in out["arms"].items() if not k.endswith("_meta")]
    out["verdict"] = {"n_ckpts": len(vals), "all_distinct": len(set(vals)) == len(vals),
                      "values": vals}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}; all_distinct={out['verdict']['all_distinct']}")


if __name__ == "__main__":
    main()
