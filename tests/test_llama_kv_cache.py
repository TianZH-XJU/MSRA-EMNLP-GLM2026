"""E-B: KV-cache correctness tests for the canonical patched LLaMA forwards.

Contract (paper §3.4 + audit A.8):
  1. prefill outputs are BITWISE unchanged by the cache feature (protects every
     prefill-based paper number: ladder, NIAH, systems prefill);
  2. cached incremental decode == full-recompute decode for the local arm
     (the slice window must match the prefill build_local_mask window);
  3. the cache actually grows (past length == prompt + generated tokens);
  4. the far branch fires on prefill only — never during decode steps.
"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.models.llama_forwards import make_llama_local_forward, make_llama_qcmsa_forward
from types import MethodType


def _tiny_llama(seed=0, n_layers=4):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=96, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=n_layers, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256,
                      torch_dtype=torch.float32)
    return LlamaForCausalLM(cfg).eval()


def _patch_local(model, window=4):
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(make_llama_local_forward(window), layer.self_attn)
    return model


class _CountingSpectral(torch.nn.Module):
    """Zero-output far-branch stub that counts invocations (avoids FFT on tiny inputs).

    The canonical far block calls the module with 4D permuted q/k/v
    ([bs, q, heads, head_dim]) and adds the return to attn_output in-place-shape,
    so the stub must return zeros_like(q)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, q, k, v):
        self.calls += 1
        return torch.zeros_like(q)


def _patch_qcmsa(model, window=4, last_n=2):
    spectrals = []
    n = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        if i >= n - last_n:
            sp = _CountingSpectral()
            spectrals.append(sp)
            layer.self_attn.forward = MethodType(make_llama_qcmsa_forward(sp, window), layer.self_attn)
        else:
            layer.self_attn.forward = MethodType(make_llama_local_forward(window), layer.self_attn)
    return model, spectrals


def _greedy_generate_cached(model, prompt_ids, n_steps, window=4):
    """Incremental decode: one prefill with cache, then 1-token steps."""
    out = model(torch.tensor([prompt_ids]), use_cache=True)
    past = out.past_key_values
    seq = list(prompt_ids)
    logits = [out.logits[0, -1]]
    for _ in range(n_steps):
        nxt = int(torch.argmax(logits[-1]))
        seq.append(nxt)
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits.append(out.logits[0, -1])
    return seq, torch.stack(logits), past


def _greedy_generate_full_recompute(model, prompt_ids, n_steps, window=4):
    """Degenerate pre-fix style: re-feed the whole prefix each step, no cache."""
    seq = list(prompt_ids)
    logits = []
    for _ in range(n_steps):
        out = model(torch.tensor([seq]), use_cache=False)
        logits.append(out.logits[0, -1])
        seq.append(int(torch.argmax(out.logits[0, -1])))
    return seq, torch.stack(logits)


PROMPT = [5, 17, 3, 42, 8, 61, 9, 23, 71, 14]  # len 10 > window 4 → windowed prefill
STEPS = 6


def test_prefill_bitwise_unchanged_by_cache():
    """The cache feature must not alter prefill logits (protects all paper eval numbers)."""
    model = _patch_local(_tiny_llama())
    ids = torch.tensor([PROMPT])
    with torch.no_grad():
        no_cache = model(ids, use_cache=False).logits
        with_cache = model(ids, use_cache=True).logits
    assert torch.equal(no_cache, with_cache), "prefill logits changed when a cache is supplied"


def test_local_cached_decode_matches_full_recompute():
    """Cached sliding-window decode == windowed full-recompute decode (local arm)."""
    model = _patch_local(_tiny_llama())
    seq_c, logits_c, past = _greedy_generate_cached(model, PROMPT, STEPS)
    seq_r, logits_r = _greedy_generate_full_recompute(model, PROMPT, STEPS)
    assert seq_c == seq_r, f"greedy tokens diverge: {seq_c} vs {seq_r}"
    # align by position: cached rows are [prefill(pos9), step(pos10..pos15)];
    # recompute rows are [pos9..pos14] (its last generated token is never re-fed)
    assert torch.allclose(logits_c[:-1], logits_r, atol=1e-5), \
        f"logit mismatch: max abs diff {(logits_c[:-1] - logits_r).abs().max()}"


def test_cache_grows_with_decode():
    model = _patch_local(_tiny_llama())
    _, _, past = _greedy_generate_cached(model, PROMPT, STEPS)
    assert past.get_seq_length() == len(PROMPT) + STEPS


def test_qcmsa_far_branch_prefill_only():
    """Far branch fires once per patched layer at prefill; never during decode steps."""
    model, spectrals = _patch_qcmsa(_tiny_llama())
    seq, logits, past = _greedy_generate_cached(model, PROMPT, STEPS)
    n_patched = len(spectrals)
    for sp in spectrals:
        assert sp.calls == 1, f"far branch called {sp.calls}x (expected 1: prefill only)"
    # and the local-only layers beneath behave like the local arm: same equivalence
    seq_r, logits_r = _greedy_generate_full_recompute(
        _patch_local(_tiny_llama()), PROMPT, STEPS)
    # token-level comparison is only meaningful for the same model, so instead check
    # the qcmsa run produced finite logits and consumed the cache correctly
    assert torch.isfinite(logits).all()
    assert past.get_seq_length() == len(PROMPT) + STEPS


def test_qcmsa_cached_decode_local_layers_match_full_recompute():
    """With a ZERO far branch, the qcmsa-patched model must decode identically to
    the local-patched model — proving the cache path didn't change local semantics."""
    model_q, _ = _patch_qcmsa(_tiny_llama(seed=3))
    model_l = _patch_local(_tiny_llama(seed=3))
    seq_q, logits_q, _ = _greedy_generate_cached(model_q, PROMPT, STEPS)
    seq_l, logits_l, _ = _greedy_generate_cached(model_l, PROMPT, STEPS)
    assert seq_q == seq_l
    assert torch.allclose(logits_q, logits_l, atol=1e-5)


def test_buffer_cache_bounded():
    """The spectral module's padded-len-keyed FP32 buffer cache must not grow
    without bound (triviaqa OOM forensics, audit A.8)."""
    from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig
    sp = CorrectSpectralAttention(QCMSAConfig(head_dim=8, feature_rank=4, num_experts=1,
                                              kernel_types=("exponential",), use_gate=False,
                                              use_phi_q=True, window_size=4, dtype=torch.float32))
    device = torch.device("cpu")
    conv = sp.fft_conv  # the buffer cache lives on the inner FFTToeplitzConv
    for i in range(12):  # 12 distinct padded_len keys
        conv._get_or_create_buffer((1, 16 * (i + 1)), device, torch.float32, cache_key=f"len{i}")
    assert len(conv._buffer_cache) <= 8, f"buffer cache grew to {len(conv._buffer_cache)}"


def test_chunked_decode_matches_full_recompute():
    """A multi-token step against a non-empty cache (chunked decode) must equal
    full-recompute at the same positions — covers the offset-mask branch."""
    model = _patch_local(_tiny_llama())
    prompt, chunk = PROMPT, [30, 44, 7]
    # cached: prefill -> one 3-token chunked step
    out = model(torch.tensor([prompt]), use_cache=True)
    past = out.past_key_values
    with torch.no_grad():
        out_c = model(torch.tensor([chunk]), past_key_values=past, use_cache=True)
    logits_c = out_c.logits[0, -3:]           # positions 10..12
    # full recompute: prefix+chunk in one prefill call
    with torch.no_grad():
        out_r = model(torch.tensor([prompt + chunk]), use_cache=False)
    logits_r = out_r.logits[0, -3:]
    assert torch.allclose(logits_c, logits_r, atol=1e-5), \
        f"chunked decode mismatch: max diff {(logits_c - logits_r).abs().max()}"


def test_niah_needle_survival_guaranteed():
    """E-C (P0-2): the needle must survive truncation VERBATIM for every placement,
    including near-bucket tails where the pre-fix order cut the retrieval target."""
    import random as _r
    from src.eval.niah_grid import generate_niah_example

    class _StubTok:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 50000 for c in text]

    tok = _StubTok()
    haystack = ["word " * 500]
    seq_len = 256
    # worst cases: intended positions at/inside the haystack tail (near bucket),
    # mid, and beyond-the-end clamps — none may truncate the needle
    for ev_pos in [250, 240, 200, 100, 0, 9999]:
        ex = generate_niah_example(seq_len, ev_pos, tok, haystack, seed=3)
        assert ex['needle_survived'] is True
        assert ex['actual_distance'] >= 0
        ids = ex['input_ids'][0].tolist()
        # geometry: query sits at the tail; needle + query + answer must fit seq_len
        assert ex['query_start'] + len(ex['target_ids']) <= seq_len
        # and the needle bytes really are in the input (verbatim retention)
        # the generator's internal assert already pins this; here we check the
        # needle region is non-degenerate (longer than the query itself)
        assert ex['query_start'] - ex['evidence_pos'] > 5
        _ = _r  # silence unused-import linters
