"""
Phase ① gate tests: forward parity (two-level criterion) + FFT causal-conv correctness.

Parity contract (PHASE1_EXPERIMENT_PLAN.md ①-0):
  patched-model(gamma:=0) vs local baseline
    - fp32 bitwise equal        -> converged (PASS)
    - non-bitwise, atol<=1e-6   -> mathematically equivalent but copies NOT converged (FAIL here;
                                   drift must be resolved, a green tick must not hide drift)
  Also asserts isfinite(y_far) before comparison: NaN from 0*inf is an overflow FINDING,
  not an equivalence failure (known fp16 random-token exp overflow).

FFT contract (①-2): out[t] = sum_l k[l] * v[t-l] via FFT == masked brute force;
  in-window rows (t <= w) are zero up to fp32 FFT noise (atol 1e-5); impulse probes at
  the boundary lags {w-1, w, w+1} land in the correct output positions.
"""

import math
import pytest
import torch
from types import MethodType

from src.models.llama_forwards import make_llama_local_forward, make_llama_qcmsa_forward
from src.kernels.correct_spectral_attention import (
    CorrectSpectralAttention, QCMSAConfig, FFTToeplitzConv, create_causal_kernel_tensor,
)


def _tiny_llama(seed=0, layers=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    return LlamaForCausalLM(cfg).eval()


def _make_spectral(window_size, device='cpu'):
    kcfg = QCMSAConfig(head_dim=16, feature_rank=8, num_experts=1, use_gate=False,
                       window_size=window_size, dtype=torch.float32)
    return CorrectSpectralAttention(kcfg).to(device)


class TestForwardParity:
    """patched(gamma:=0) must equal local: bitwise in fp32."""

    def test_patched_gamma0_bitwise_equals_local(self):
        model_local = _tiny_llama(seed=0)
        model_qc = _tiny_llama(seed=0)  # identical weights
        window = 16
        n_layer = len(model_qc.model.layers)

        for layer in model_local.model.layers:
            layer.self_attn.forward = MethodType(make_llama_local_forward(window), layer.self_attn)

        for i, layer in enumerate(model_qc.model.layers):
            if i >= n_layer - 1:  # last layer patched
                sp = _make_spectral(window)
                with torch.no_grad():
                    sp.gamma.zero_()          # gamma := 0
                layer.self_attn.qcmsa_spectral = sp
                layer.self_attn.forward = MethodType(make_llama_qcmsa_forward(sp, window), layer.self_attn)
            else:
                layer.self_attn.forward = MethodType(make_llama_local_forward(window), layer.self_attn)

        torch.manual_seed(1)
        ids = torch.randint(10, 100, (1, 64))

        with torch.no_grad():
            out_local = model_local(ids).logits
            out_qc = model_qc(ids).logits

        assert torch.isfinite(out_qc).all(), "patched output non-finite (NaN trap: check y_far overflow)"
        max_abs = (out_local - out_qc).abs().max().item()
        bitwise = torch.equal(out_local, out_qc)
        assert bitwise, (
            f"non-bitwise (max|diff|={max_abs:.2e}): fp32 bitwise contract violated — "
            "either copy drift or far-branch leakage at gamma:=0. "
            "A tolerance pass would HIDE drift; do not weaken this to atol."
        )

    def test_patched_with_gamma_untouched_differs(self):
        """Sanity: with gamma at init (1e-5) the patched model may differ slightly;
        with a large gamma it must differ clearly (far branch is live)."""
        model_local = _tiny_llama(seed=0)
        model_qc = _tiny_llama(seed=0)
        window = 16
        n_layer = len(model_qc.model.layers)
        for layer in model_local.model.layers:
            layer.self_attn.forward = MethodType(make_llama_local_forward(window), layer.self_attn)
        sp = _make_spectral(window)
        with torch.no_grad():
            sp.gamma.fill_(0.5)  # strong far branch
            layer = model_qc.model.layers[n_layer - 1]
        layer.self_attn.qcmsa_spectral = sp
        layer.self_attn.forward = MethodType(make_llama_qcmsa_forward(sp, window), layer.self_attn)
        torch.manual_seed(1)
        ids = torch.randint(10, 100, (1, 64))
        with torch.no_grad():
            d = (model_local(ids).logits - model_qc(ids).logits).abs().max().item()
        assert d > 1e-4, "far branch with gamma=0.5 had no effect — far path is dead"

    def test_inwindow_far_branch_is_zero(self):
        """F(i) is empty for query positions <= w: y_far must be exactly 0 there
        (up to fp32 FFT noise), for ANY sigma and ANY inputs."""
        sp = _make_spectral(window_size=32)
        torch.manual_seed(3)
        q = torch.randn(1, 128, 4, 16)
        k = torch.randn(1, 128, 4, 16)
        v = torch.randn(1, 128, 4, 16)
        with torch.no_grad():
            y = sp(q, k, v)
        in_window = y[:, :33]  # positions 0..32 (w=32)
        assert torch.isfinite(y).all()
        assert in_window.abs().max().item() < 1e-5, (
            f"in-window |y_far| = {in_window.abs().max().item():.2e} — exceeds fp32 FFT noise; "
            "suspect far-mask off-by-one, LN-bias on empty-F path, or cache pollution"
        )


class TestFFTConvCausal:
    """FFT conv == masked brute force; boundary lags land correctly."""

    @staticmethod
    def _brute(kernel, value):
        T = value.shape[1]
        out = torch.zeros_like(value)
        for t in range(T):
            for l in range(0, min(t + 1, kernel.shape[0])):
                out[:, t] += kernel[l] * value[:, t - l]
        return out

    @pytest.mark.parametrize("n", [8, 33, 100])
    def test_fft_matches_brute_force(self, n):
        w = 16
        conv = FFTToeplitzConv()
        torch.manual_seed(n)
        value = torch.randn(2, n, 4, 8)
        kernel = create_causal_kernel_tensor("exponential", n, torch.tensor(50.0), w, "cpu", torch.float32)
        with torch.no_grad():
            out = conv(kernel, value, n, key_hint=50.0)
            ref = self._brute(kernel, value)
        assert (out - ref).abs().max().item() < 1e-5

    def test_boundary_lags(self):
        """Far-mask semantics on impulse kernels: lags {w-1, w} must be zeroed,
        lag {w+1} must survive — off-by-one in create_causal_kernel_tensor shows here.
        (The mask is applied as in production: kernel * (positions > window_size).)"""
        w, n = 16, 64
        for lag, expect_nonzero in [(w - 1, False), (w, False), (w + 1, True)]:
            kernel = torch.zeros(n)
            kernel[lag] = 1.0
            positions = torch.arange(n)
            kernel = kernel * (positions > w).float()  # production mask
            conv = FFTToeplitzConv()
            value = torch.randn(1, n, 1, 1)
            with torch.no_grad():
                out = conv(kernel, value, n, key_hint=float(lag))
            hit = out[:, lag].abs().max().item()  # impulse at lag L arrives at query L (from key 0)
            if expect_nonzero:
                assert hit > 1e-3, "lag w+1 impulse was killed (far mask over-masks)"
            else:
                assert hit < 1e-5, f"lag {lag} impulse leaked through far mask (off-by-one)"

    def test_far_mask_boundary_in_kernel_factory(self):
        """create_causal_kernel_tensor: kernel[0..w] exactly 0, kernel[w+1] > 0."""
        n, w = 64, 16
        kernel = create_causal_kernel_tensor("exponential", n, torch.tensor(50.0), w, "cpu", torch.float32)
        assert kernel[:w + 1].abs().max().item() == 0.0, "far mask leaves mass at lag <= w (off-by-one)"
        assert kernel[w + 1].item() > 0, "far mask over-masks lag w+1"

    def test_training_no_cache_sigma_learns(self):
        """During grad-enabled steps the kernel FFT must NOT be cached, so sigma
        receives gradient (0a regression test for the cache-key collision bug)."""
        cfg = QCMSAConfig(head_dim=16, feature_rank=8, num_experts=1, use_gate=False,
                          window_size=8, dtype=torch.float32)
        m = CorrectSpectralAttention(cfg)
        q = torch.randn(1, 32, 2, 16)
        k = torch.randn(1, 32, 2, 16)
        v = torch.randn(1, 32, 2, 16)
        init = m.log_sigma.detach().clone()
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        for _ in range(20):
            opt.zero_grad()
            out = m(q, k, v)
            (-((out - 0.3) ** 2).mean() * 1e4).backward()
            opt.step()
        assert len(m.fft_conv._kernel_fft_cache) == 0, "kernel FFT cached during training"
        assert (m.log_sigma.detach() - init).abs().max().item() > 0, "sigma frozen (cache collision regression)"

    def test_inference_cache_discriminates_sigma(self):
        conv = FFTToeplitzConv()
        n, w = 128, 16
        v = torch.randn(1, n, 2, 4)
        with torch.no_grad():
            o1 = conv(create_causal_kernel_tensor("exponential", n, torch.tensor(10.0), w, "cpu", torch.float32), v, n, key_hint=10.0)
            o2 = conv(create_causal_kernel_tensor("exponential", n, torch.tensor(100.0), w, "cpu", torch.float32), v, n, key_hint=100.0)
        assert len(conv._kernel_fft_cache) == 2, "different sigmas share one cache entry (collision regression)"
        assert (o1 - o2).abs().max().item() > 0, "sigma has no effect on forward (cache collision regression)"
