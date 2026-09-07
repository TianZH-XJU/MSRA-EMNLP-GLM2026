"""
Canonical LLaMA attention forwards — single source of truth (Phase ①-0 convergence).

History: five drifting copies of these builders lived in llama_train.py, llama_loader.py,
systems_benchmark.py, llama_per_position.py and qcmsa_llama.py. Copy drift is a proven
failure mode in this project, so all callers now import from here.

Parity contract (tests/test_forward_parity.py):
  patched-model(gamma:=0) vs local baseline  →  fp32 bitwise equal;
  non-bitwise but atol<=1e-5 means "equivalent but NOT converged" and must block release.
"""

import torch
from types import MethodType


def build_local_mask(q_len, kv_len, window_size, device, dtype):
    """Additive causal + sliding-window mask: 0 where allowed, -inf where forbidden."""
    q_idx = torch.arange(q_len, device=device)
    k_idx = torch.arange(kv_len, device=device)
    causal = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)
    local = (q_idx.unsqueeze(1) - k_idx.unsqueeze(0)) <= window_size
    mask = torch.where(causal & local, 0.0, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0).to(dtype)


def _llama_attention_forward(self, hidden_states, position_embeddings,
                             attention_mask, past_key_values, spectral_module, window_size, **kwargs):
    """Shared body: QKV + RoPE + local-mask eager attention (+ optional far branch).

    KV-cache semantics (E-B fix, 2026-09-07):
      - prefill (empty cache): unchanged bitwise — windowed additive mask when the
        prompt exceeds the window, full causal otherwise; the far branch fires here
        and only here. k/v are written into the cache when one is supplied; on an
        empty cache update() is a value-identity, so prefill outputs are
        bit-identical to the pre-fix code (test-pinned).
      - incremental decode (non-empty cache): the current token attends the last
        window_size+1 cached keys — a true sliding-window decode; the far branch is
        structurally OFF (prefill-only), which is what §3.4 of the paper claims.
        mask=None here assumes an unpadded batch (bs=1, the generate case); a padded
        batch would attend pad tokens and must pass a real mask instead.
      - multi-token step against a non-empty cache (chunked decode): offset
        window+causal mask over the full cache.
    The pre-fix behavior — cache accepted but never written, so HF generate() sliced
    each step's input to the last token and every generated token attended only
    itself — is documented in audit entry A.8 of the paper.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    q_len = query_states.shape[2]
    if past_key_values is not None:
        # HF-idiomatic: update() returns the FULL concatenated k/v (past + current);
        # all attention below must use this return, not the fresh projections alone
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    kv_len = key_states.shape[2]

    # prefill detection must be per-layer (q_len vs THIS layer's post-update kv_len):
    # DynamicCache.get_seq_length() is global across layers, so a past_len==0 test
    # would misclassify every layer after the first as decode during prefill
    is_prefill = (q_len == kv_len)
    far_active = is_prefill and window_size < q_len

    if is_prefill:
        if far_active:
            local_mask = build_local_mask(q_len, kv_len, window_size,
                                          query_states.device, query_states.dtype)
        else:
            local_mask = attention_mask
        k_att, v_att = key_states, value_states
    elif q_len == 1:
        # incremental sliding-window decode: only the last (window+1) keys are
        # visible; causality is implicit since the cache holds past + current only
        w_eff = min(window_size + 1, kv_len)
        k_att = key_states[:, :, -w_eff:, :]
        v_att = value_states[:, :, -w_eff:, :]
        local_mask = None
    else:
        # chunked decode against a non-empty cache: queries sit at global
        # positions [kv_len-q_len, kv_len); rebuild the offset window mask
        q_idx = torch.arange(kv_len - q_len, kv_len, device=query_states.device)
        k_idx = torch.arange(kv_len, device=query_states.device)
        causal = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)
        local = (q_idx.unsqueeze(1) - k_idx.unsqueeze(0)) <= window_size
        local_mask = torch.where(causal & local, 0.0, float('-inf'))
        local_mask = local_mask.unsqueeze(0).unsqueeze(0).to(query_states.dtype)
        k_att, v_att = key_states, value_states

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, k_att, v_att, local_mask,
        dropout=0.0, scaling=self.scaling, **kwargs,
    )

    # Far branch: prefill only, FP32 under disabled autocast, added before o_proj
    if spectral_module is not None and far_active:
        q_s = query_states.permute(0, 2, 1, 3)
        k_s = key_states.permute(0, 2, 1, 3)
        v_s = value_states.permute(0, 2, 1, 3)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            y_far = spectral_module(q_s.float(), k_s.float(), v_s.float())
        assert torch.isfinite(y_far).all(), "far branch produced non-finite values"
        attn_output = attn_output + y_far.to(attn_output.dtype)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def make_llama_qcmsa_forward(spectral_module, window_size):
    """LLaMA attention with local mask + MSRA far branch (canonical)."""
    def qcmsa_forward(self, hidden_states, position_embeddings=None,
                      attention_mask=None, past_key_values=None, **kwargs):
        return _llama_attention_forward(self, hidden_states, position_embeddings,
                                        attention_mask, past_key_values,
                                        spectral_module, window_size, **kwargs)
    return qcmsa_forward


def make_llama_local_forward(window_size):
    """LLaMA attention with local mask only, no far branch (canonical)."""
    def local_forward(self, hidden_states, position_embeddings=None,
                      attention_mask=None, past_key_values=None, **kwargs):
        return _llama_attention_forward(self, hidden_states, position_embeddings,
                                        attention_mask, past_key_values,
                                        None, window_size, **kwargs)
    return local_forward


def patch_llama_layers(model, window_size, patch_last_n, spectral_factory, ckpt=None, device='cuda'):
    """Patch the last `patch_last_n` layers with the qcmsa forward (+ spectral module),
    earlier layers with the local forward. `spectral_factory()` -> CorrectSpectralAttention.
    If `ckpt` (dict layer-key -> state_dict) is given, weights are loaded and the
    module's kernel-FFT cache is cleared (stale-cache defense)."""
    n_layer = len(model.model.layers)
    start_layer = n_layer - min(patch_last_n, n_layer)
    for i, layer in enumerate(model.model.layers):
        if i >= start_layer:
            sp = spectral_factory().to(device=device, dtype=torch.float32)
            layer.self_attn.qcmsa_spectral = sp
            if ckpt is not None and str(i) in ckpt:
                sd = {k: v.to(device) for k, v in ckpt[str(i)].items()}
                # strict=True: a silent key mismatch here would leave random spectral
                # weights in place — the exact silent-no-op failure mode of audit A.6
                sp.load_state_dict(sd, strict=True)
                sp.clear_caches()
            layer.self_attn.forward = MethodType(make_llama_qcmsa_forward(sp, window_size), layer.self_attn)
        else:
            layer.self_attn.forward = MethodType(make_llama_local_forward(window_size), layer.self_attn)
    return model
