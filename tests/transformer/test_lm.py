import torch

from transformerlm.transformer.transformer_lm import TransformerLM


def make_tiny_model(device):
    return TransformerLM(
        vocab_size=16,
        context_length=8,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10_000.0,
        device=device,
        dtype=torch.float32,
    )


def test_forward_shape(device):
    model = make_tiny_model(device)
    B, T = 2, 4
    x = torch.randint(0, 16, (B, T), device=device)
    logits = model(x)
    assert logits.shape == (B, T, 16)


def test_decode_grows_by_steps_without_eos(device):
    model = make_tiny_model(device)
    B, T0 = 1, 3
    x0 = torch.randint(0, 16, (B, T0), device=device)

    steps = 5
    out = model.decode(in_indices=x0.clone(), context_length=steps, temperature=1.0, p=0.8, eos_token_id=None)
    assert out.shape[1] == T0 + steps


def test_decode_deterministic_with_p_zero(device):
    model = make_tiny_model(device)
    B, T0 = 1, 3
    x0 = torch.randint(0, 16, (B, T0), device=device)

    steps = 4
    out1 = model.decode(in_indices=x0.clone(), context_length=steps, temperature=1.0, p=0.0, eos_token_id=None)
    out2 = model.decode(in_indices=x0.clone(), context_length=steps, temperature=1.0, p=0.0, eos_token_id=None)
    # With p=0, sampling reduces to argmax → deterministic
    assert torch.equal(out1, out2)

