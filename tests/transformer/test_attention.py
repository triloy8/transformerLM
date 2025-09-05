import torch
import pytest

from transformerlm.transformer.transformer_lm import MultiheadSelfAttentionRoPE, softmax


def test_causal_mask_invariance(device):
    d_model = 8
    num_heads = 2
    max_seq_len = 8
    theta = 10_000.0
    attn = MultiheadSelfAttentionRoPE(d_model=d_model,
                                      num_heads=num_heads,
                                      max_seq_len=max_seq_len,
                                      theta=theta,
                                      device=device)

    T = 4
    x = torch.randn(1, T, d_model, device=device)
    tok_pos = torch.arange(T, dtype=torch.long, device=device)

    out1 = attn(x, tok_pos)

    # Modify future tokens (> i) and confirm outputs up to i unchanged
    i = 1  # check positions 0..i
    x2 = x.clone()
    x2[:, i + 1 :, :] += 1.0
    out2 = attn(x2, tok_pos)

    assert torch.allclose(out1[:, : i + 1, :], out2[:, : i + 1, :], atol=1e-6, rtol=1e-5)


def test_softmax_matches_torch(device):
    logits = torch.tensor([[1000.0, 0.0, -1000.0], [0.2, -0.3, 0.1]], device=device)
    sx = softmax(logits, dim=-1)
    tx = torch.nn.functional.softmax(logits, dim=-1)
    assert torch.allclose(sx, tx, atol=1e-7, rtol=1e-6)