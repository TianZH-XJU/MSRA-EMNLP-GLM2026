"""
Shared LLaMA eval loader — drop-in model loading for all eval scripts.

Provides clean API for loading LLaMA with full/local/QC-MSA configurations.
Reuses forward functions from systems_benchmark.py.

Usage:
  from src.eval.llama_loader import load_llama_full, load_llama_local, load_llama_qcmsa
  model = load_llama_local(model_path, window_size=128)
  model = load_llama_qcmsa(model_path, window_size=128, ckpt_path="results/llama_512_s2.pt")
"""

import torch, gc, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import AutoModelForCausalLM
from types import MethodType
from src.models.llama_forwards import make_llama_qcmsa_forward as _make_llama_qcmsa_forward, make_llama_local_forward as _make_llama_local_forward


def load_llama_full(model_path, device='cuda', dtype=torch.float16):
    """Load LLaMA with full attention (no patching)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype).to(device).eval()
    return model


def load_llama_local(model_path, window_size=128, device='cuda', dtype=torch.float16):
    """Load LLaMA with custom local sliding window attention."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(_make_llama_local_forward(window_size), layer.self_attn)
    return model


def load_llama_qcmsa(model_path, window_size=128, ckpt_path=None, device='cuda',
                     patch_layers=8, feature_rank=16, num_experts=4, dtype=torch.float16):
    """Load LLaMA with QC-MSA (local mask + spectral far branch).
    Infers M / gate from the checkpoint when one is given (provenance-safe loading)."""
    from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager").to(device).eval()

    ckpt = None
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd0 = next(iter(ckpt.values()))
        num_experts = len(sd0["log_sigma"])          # infer M from ckpt (M=1 vs M=4 ckpts)
        use_gate = "gate.weight" in sd0
    else:
        use_gate = True

    n_layer = len(model.model.layers)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    patch_n = min(patch_layers, n_layer)
    start_layer = n_layer - patch_n

    kcfg = QCMSAConfig(head_dim=head_dim, feature_rank=feature_rank, num_experts=num_experts,
                       kernel_types=tuple(["exponential"] * num_experts),
                       use_gate=use_gate, window_size=window_size, dtype=torch.float32)

    for i, layer in enumerate(model.model.layers):
        if i >= start_layer:
            sp = CorrectSpectralAttention(kcfg).to(device=device, dtype=torch.float32)
            sp.log_sigma.data = torch.tensor([3.0, 4.0, 5.0, 6.0][:num_experts], device=device)
            layer.self_attn.qcmsa_spectral = sp
            layer.self_attn.forward = MethodType(_make_llama_qcmsa_forward(sp, window_size), layer.self_attn)
        else:
            layer.self_attn.forward = MethodType(_make_llama_local_forward(window_size), layer.self_attn)

    # Load checkpoint if provided
    if ckpt is not None:
        loaded = 0
        for i, layer in enumerate(model.model.layers):
            if i >= start_layer:
                key = str(i)
                if key in ckpt:
                    sd = {k: v.to(device) for k, v in ckpt[key].items()}
                    layer.self_attn.qcmsa_spectral.load_state_dict(sd, strict=False)
                    layer.self_attn.qcmsa_spectral.clear_caches()
                    loaded += 1
        print(f"  Loaded QC-MSA checkpoint: {loaded}/{patch_n} layers (M={num_experts}, gate={use_gate})")

    return model
