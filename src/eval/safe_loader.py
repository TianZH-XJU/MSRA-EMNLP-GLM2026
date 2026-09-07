"""
Safe model loader using custom eager masks — bypasses HF sliding window CUDA bug.

Provides drop-in replacements for load_gpt2_lm that use raw HF models
with custom causal sliding window masks (eager attention, no HF masking bug).

Key fix: patches BOTH GPT2Model.forward (to avoid create_causal_mask) AND
GPT2Attention.forward (for custom local mask).

Usage:
  from src.eval.safe_loader import load_full, load_local, load_qcmsa
  model = load_full(model_path)       # Full attention (no sliding window)
  model = load_local(model_path, 128) # Local ws=128 (custom causal mask)
  model = load_qcmsa(model_path, 128) # Local + QC-MSA far branch
"""

import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from types import MethodType


def _build_additive_causal_mask(seq_len, window_size, device, dtype):
    """Build additive causal mask: 0 for allowed, -inf for forbidden.
    Done on CPU first, then moved to device to avoid CUDA asserts.
    """
    q_idx = torch.arange(seq_len)
    k_idx = torch.arange(seq_len)
    causal = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)
    if window_size >= 0:
        local = (q_idx.unsqueeze(1) - k_idx.unsqueeze(0)) <= window_size
        allow = causal & local
    else:
        allow = causal
    mask = torch.where(allow, 0.0, float('-inf'))
    # (1, 1, seq_len, seq_len) — broadcastable over batch and heads
    mask = mask.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    return mask


def _make_transformer_forward(window_size):
    """Build a GPT2Model.forward that avoids HF create_causal_mask (CUDA bug)."""

    def transformer_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        if use_cache is None:
            use_cache = self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("Must specify input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        # Build mask manually — no HF masking utilities (they have CUDA bug)
        seq_len = inputs_embeds.shape[1]
        causal_mask = _build_additive_causal_mask(
            seq_len, window_size, inputs_embeds.device, inputs_embeds.dtype
        )

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        for block in self.h:
            hidden_states = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                causal_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        past_key_values = past_key_values if use_cache else None

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    return transformer_forward


def make_gpt2_local_forward(window_size, device='cuda'):
    """GPT-2 local attention via eager + custom causal sliding window mask."""
    def local_forward(self, hidden_states, past_key_values=None, attention_mask=None, **kwargs):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        batch, seq_len = hidden_states.shape[:2]
        nh, hd = self.num_heads, self.head_dim
        shape = (batch, seq_len, nh, hd)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)

        # Causal + local mask
        q_idx = torch.arange(seq_len, device=query.device)
        k_idx = torch.arange(seq_len, device=key.device)
        causal = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)
        local = (q_idx.unsqueeze(1) - k_idx.unsqueeze(0)) <= window_size
        mask = torch.where(causal & local, 0.0, float('-inf'))
        mask = mask.unsqueeze(0).unsqueeze(0).to(query.dtype)

        from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward
        attn_output, _ = eager_attention_forward(
            self, query, key, value, mask, dropout=0.0, scaling=self.scaling)
        attn_output = attn_output.reshape(batch, seq_len, nh * hd).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, None
    return local_forward


def load_full(model_path, dtype=torch.float16):
    """Load GPT-2 with full attention (raw HF model, no patching)."""
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    return model


def load_local(model_path, window_size=128, dtype=torch.float16):
    """Load GPT-2 with local attention via custom eager mask (NO HF bug)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation='eager')

    # Patch transformer forward to avoid HF masking utils (CUDA bug)
    model.transformer.forward = MethodType(
        _make_transformer_forward(window_size), model.transformer)

    # Patch each attention block
    for blk in model.transformer.h:
        blk.attn.forward = MethodType(make_gpt2_local_forward(window_size), blk.attn)
    model.eval()
    return model


def load_qcmsa(model_path, window_size=128, feature_rank=16, num_experts=4,
               patch_last_n=12, dtype=torch.float16, use_phi_q=False):
    """Load GPT-2 with QC-MSA (local custom mask + spectral far branch)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.kernels.correct_spectral_attention import CorrectSpectralAttention, QCMSAConfig

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation='eager')

    # Patch transformer forward to avoid HF masking utils
    model.transformer.forward = MethodType(
        _make_transformer_forward(window_size), model.transformer)

    device = next(model.parameters()).device
    n_layer = model.config.n_layer
    start_layer = n_layer - min(patch_last_n, n_layer)

    kcfg = QCMSAConfig(
        head_dim=model.config.n_embd // model.config.n_head,
        feature_rank=feature_rank, num_experts=num_experts,
        kernel_types=tuple(["exponential"] * num_experts),
        use_gate=True, use_phi_q=use_phi_q,
        window_size=window_size, dtype=torch.float32,
    )

    for i, blk in enumerate(model.transformer.h):
        if i >= start_layer:
            sp = CorrectSpectralAttention(kcfg).to(device=device, dtype=torch.float32)
            sp.log_sigma.data = torch.tensor([3.0, 4.0, 5.0, 6.0][:num_experts], device=device)
            blk.attn.qcmsa_spectral = sp
            blk.qcmsa_spectral = sp

            # QC-MSA forward: local mask + far branch
            def make_qcmsa_forward(window_size, sp_module):
                def qcmsa_forward(self, hidden_states, past_key_values=None,
                                  attention_mask=None, **kwargs):
                    query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
                    batch, seq_len = hidden_states.shape[:2]
                    nh, hd = self.num_heads, self.head_dim
                    shape = (batch, seq_len, nh, hd)
                    query = query.view(shape).transpose(1, 2)
                    key = key.view(shape).transpose(1, 2)
                    value = value.view(shape).transpose(1, 2)

                    q_idx = torch.arange(seq_len, device=query.device)
                    k_idx = torch.arange(seq_len, device=key.device)
                    causal = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)
                    local = (q_idx.unsqueeze(1) - k_idx.unsqueeze(0)) <= window_size
                    mask = torch.where(causal & local, 0.0, float('-inf'))
                    mask = mask.unsqueeze(0).unsqueeze(0).to(query.dtype)

                    from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward
                    attn_output, _ = eager_attention_forward(
                        self, query, key, value, mask, dropout=0.0, scaling=self.scaling)

                    q_len = query.shape[2]; kv_len = key.shape[2]
                    if q_len == kv_len and window_size < q_len:
                        q_s = query.permute(0,2,1,3); k_s = key.permute(0,2,1,3); v_s = value.permute(0,2,1,3)
                        with torch.amp.autocast(device_type='cuda', enabled=False):
                            y_far = sp_module(q_s.float(), k_s.float(), v_s.float())
                        attn_output = attn_output + y_far.to(attn_output.dtype)

                    attn_output = attn_output.reshape(batch, seq_len, nh*hd).contiguous()
                    attn_output = self.c_proj(attn_output)
                    attn_output = self.resid_dropout(attn_output)
                    return attn_output, None
                return qcmsa_forward

            blk.attn.forward = MethodType(make_qcmsa_forward(window_size, sp), blk.attn)
        else:
            blk.attn.forward = MethodType(make_gpt2_local_forward(window_size), blk.attn)
            blk.attn.qcmsa_spectral = None
            blk.qcmsa_spectral = None

    model.eval()
    return model
