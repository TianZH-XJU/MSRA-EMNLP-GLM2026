"""
② fixed-split re-evaluation of all checkpoints (pure inference, PHASE1_EXPERIMENT_PLAN step ②).

Evaluates local / MSRA(ckpt) / full on the SAME fixed manifest chunks — removes the
per-seed eval-draw variance that caused cross-table gap-closed disagreements (P0-B).
Provenance (manifest sha256 + ckpt hash + code hash) is recorded; every arm shares the
same chunks, so gap-closed numbers are directly comparable.

Usage:
  CUDA_VISIBLE_DEVICES=5 python -m src.eval.eval_fixed_split \
      --manifest results/eval_split_256.json --n_chunks 16 \
      --ckpts results/final_M1_L8_s2.pt results/R_M1L8_s2.pt ... --out results/fixed_split_eval.json
"""

import argparse, hashlib, gc, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig
from src.models.llama_forwards import patch_llama_layers, make_llama_local_forward
from types import MethodType

MP = "/data1/zhoujun/Auto-claude-code-research-in-sleep/basemodel/llama3.2-1b"
PG19 = "/data1/zhoujun/Auto-claude-code-research-in-sleep/dataset/pg19"


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def load_chunks(tok, manifest_path, L, n_chunks):
    man = json.load(open(manifest_path))
    import pandas as pd
    books = []
    for f in sorted(set(b["file"] for b in man["books"])):
        df = pd.read_parquet(os.path.join(PG19, "data", f))
        for b in [x for x in man["books"] if x["file"] == f]:
            books.append(tok.encode(df["text"].tolist()[b["row"]],
                                    add_special_tokens=False, truncation=True, max_length=8192))
    chunks = []
    for toks in books:
        if len(toks) >= L + 1:
            chunks.extend(toks[i:i + L] for i in range(0, len(toks) - L, L))
    return chunks[:n_chunks], man["sha256"]


@torch.no_grad()
def ppl(model, chunks, device):
    model.eval()
    tl, tt = 0.0, 0
    for c in chunks:
        inp = torch.tensor([c], device=device)
        out = model(inp)
        tl += F.cross_entropy(out.logits[0, :-1], torch.tensor(c[1:], device=device), reduction='sum').item()
        tt += len(c) - 1
    return torch.exp(torch.tensor(tl / tt)).item() if tt else float('nan')


# thread cap: torch defaults to 128 OMP threads on this 256-core host; concurrent
# background jobs then oversubscribe CPU to >5000% (2026-09-07 ops incident)
torch.set_num_threads(8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/eval_split_256.json")
    ap.add_argument("--n_chunks", type=int, default=16)
    ap.add_argument("--lengths", default="512")
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--out", default="results/fixed_split_eval.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device
    lengths = [int(x) for x in args.lengths.split(",")]

    tok = AutoTokenizer.from_pretrained(MP)
    results = {"provenance": {"manifest": args.manifest,
                              "manifest_sha256": json.load(open(args.manifest))["sha256"],
                              "code_sha256": sha16(os.path.join(os.path.dirname(__file__), '..', 'kernels', 'correct_spectral_attention.py')),
                              "n_chunks": args.n_chunks, "lengths": lengths},
               "arms": {}}

    # shared chunks per length (identical across ALL arms — the whole point)
    chunk_sets = {}
    for L in lengths:
        chunks, man_hash = load_chunks(tok, args.manifest, L, args.n_chunks)
        assert len(chunks) >= 8, f"manifest yielded only {len(chunks)} chunks @{L}"
        chunk_sets[L] = chunks
    print(f"fixed chunks: { {L: len(c) for L, c in chunk_sets.items()} }")

    # full + local once (no ckpt)
    m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, attn_implementation="eager").to(device).eval()
    results["arms"]["full"] = {str(L): ppl(m, chunk_sets[L], device) for L in lengths}
    print("full:", results["arms"]["full"])
    del m; gc.collect(); torch.cuda.empty_cache()

    def add_local():
        m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, attn_implementation="eager").to(device).eval()
        from src.models.llama_forwards import make_llama_local_forward
        from types import MethodType
        for layer in m.model.layers:
            layer.self_attn.forward = MethodType(make_llama_local_forward(128), layer.self_attn)
        r = {str(L): ppl(m, chunk_sets[L], device) for L in lengths}
        del m; gc.collect(); torch.cuda.empty_cache()
        return r

    results["arms"]["local"] = add_local()
    print("local:", results["arms"]["local"])

    for ck in args.ckpts:
        if ck == "random":
            # ④ frozen-kernel control: untrained far branch (random init, gamma=1e-5), zero training
            m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, attn_implementation="eager").to(device).eval()
            def rfactory():
                sp = CorrectSpectralAttention(QCMSAConfig(head_dim=64, feature_rank=16, num_experts=1,
                    kernel_types=("exponential",), use_gate=False, use_phi_q=True, window_size=128, dtype=torch.float32))
                sp.log_sigma.data = torch.tensor([3.0])  # A6: align with training init (sigma=20.1)
                return sp
            patch_llama_layers(m, 128, patch_last_n=8, spectral_factory=rfactory, device=device)
            results["arms"]["random_untrained"] = {str(L): ppl(m, chunk_sets[L], device) for L in lengths}
            results["arms"]["random_untrained_meta"] = {"ckpt_sha256": "untrained-random-init", "n_patched_layers": 8}
            print(f"random_untrained: {results['arms']['random_untrained']}")
            del m; gc.collect(); torch.cuda.empty_cache()
            continue
        if ck.startswith("baseline:"):
            # F1: matched-baseline arms (LoRA/dwconv o_proj wrappers) at canonical 16-chunk caliber
            bl = ck.split(":")[1]
            from src.eval.matched_baselines import LoRAR1OWrapper, CausalDWConvWrapper
            m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, attn_implementation="eager").to(device).eval()
            start_bl = len(m.model.layers) - 8
            trainables = []
            for li, layer in enumerate(m.model.layers):
                layer.self_attn.forward = MethodType(make_llama_local_forward(128), layer.self_attn)
                if li >= start_bl:
                    o = layer.self_attn.o_proj
                    if bl == "lora":
                        wrap = LoRAR1OWrapper(o, m.config.hidden_size).to(device=device)
                        wrap.lora_A = wrap.lora_A.to(dtype=torch.float32)
                        wrap.lora_B = wrap.lora_B.to(dtype=torch.float32)
                    else:
                        wrap = CausalDWConvWrapper(o, m.config.hidden_size, kernel=8).to(device=device)
                        wrap.conv = wrap.conv.to(dtype=torch.float32)
                    bname = os.path.splitext(os.path.basename(ck))[0]
                    bck = torch.load(f"results/{bname}.pt", map_location=device)
                    bsd = bck[str(li)]["o_proj_wrap"]
                    missing, unexpected = wrap.load_state_dict(bsd, strict=False)
                    layer.self_attn.o_proj = wrap  # wire into the forward path (training side does the same)
                    trainables += [p for p in wrap.parameters() if p.requires_grad]
                    results.setdefault("load_check", {})[f"{bname}_L{li}"] = {"missing": missing, "unexpected": unexpected}
            # positive control: adapter must be in the forward path — otherwise this arm
            # silently evaluates as plain local (observed on first execution of this branch)
            for li in range(start_bl, len(m.model.layers)):
                assert isinstance(m.model.layers[li].self_attn.o_proj, (LoRAR1OWrapper, CausalDWConvWrapper)), \
                    f"baseline wiring broken: layer {li} o_proj is {type(m.model.layers[li].self_attn.o_proj)}"
            results["arms"][f"baseline_{bl}"] = {str(L): ppl(m, chunk_sets[L], device) for L in lengths}
            print(f"baseline_{bl}: {results['arms'][f'baseline_{bl}']}")
            del m; gc.collect(); torch.cuda.empty_cache()
            continue
        if not os.path.exists(ck):
            print(f"SKIP missing {ck}")
            continue
        name = os.path.splitext(os.path.basename(ck))[0]
        m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, attn_implementation="eager").to(device).eval()
        ckd = torch.load(ck, map_location=device)
        n_layers_patched = len(ckd)
        # A3: M inferred by width (llama_loader-correct form); any M works, not just 1/4
        M = len(next(iter(ckd.values()))["log_sigma"])
        gate = "gate.weight" in next(iter(ckd.values()))
        def factory():
            return CorrectSpectralAttention(QCMSAConfig(
                head_dim=64, feature_rank=16, num_experts=M, kernel_types=("exponential",) * M,
                use_gate=gate, use_phi_q=True, window_size=128, dtype=torch.float32))
        patch_llama_layers(m, 128, patch_last_n=n_layers_patched, spectral_factory=factory, ckpt=ckd, device=device)
        # A4: load-fidelity assertion — post-load weights must match the ckpt (catches silent load failures)
        for li, layer in enumerate(m.model.layers):
            sp = getattr(layer.self_attn, 'qcmsa_spectral', None)
            if sp is None or str(li) not in ckd: continue
            for pk, pv in ckd[str(li)].items():
                loaded = dict(sp.named_parameters())[pk] if pk in dict(sp.named_parameters()) else None
                if loaded is not None and not torch.allclose(loaded.detach().float().cpu(), pv.float().cpu(), atol=1e-6):
                    raise RuntimeError(f"load fidelity violated: {name} layer {li} {pk}")
        results["arms"][name] = {str(L): ppl(m, chunk_sets[L], device) for L in lengths}
        results["arms"][name + "_meta"] = {"ckpt_sha256": sha16(ck), "n_patched_layers": n_layers_patched}
        print(f"{name}: {results['arms'][name]}")
        del m; gc.collect(); torch.cuda.empty_cache()

    json.dump(results, open(args.out, 'w'), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
