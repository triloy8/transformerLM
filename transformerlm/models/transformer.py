import torch
import torch.nn as nn

from transformerlm.models.layers import Embedding, RMSNorm, SwiGLU, Linear
from transformerlm.models.attention import MultiheadSelfAttentionRoPE
from transformerlm.inference.sampling import softmax, top_p_filter


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.attn = MultiheadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.long)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, context_length, rope_theta, d_ff, device, dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        output_seq = self.token_embeddings(in_indices)
        for layer in self.layers:
            output_seq = layer(output_seq)
        normed_output_seq = self.ln_final(output_seq)
        logits = self.lm_head(normed_output_seq)
        return logits

    @torch.no_grad()
    def decode(self, in_indices: torch.Tensor, context_length: int | None = None, temperature=1.0, p=0.0, eos_token_id=None):
        batch_size = in_indices.shape[0]
        device = in_indices.device
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        ctx = self.context_length if context_length is None else context_length

        for _ in range(ctx):
            context_indices = in_indices if in_indices.shape[1] <= ctx else in_indices[:, -ctx:]
            logits = self(context_indices)
            logits = logits[:, -1, :] / temperature
            q = softmax(logits, dim=-1)
            filtered = top_p_filter(q, p)
            index_next = torch.multinomial(filtered, num_samples=1)
            if eos_token_id is not None:
                index_next[finished] = eos_token_id
            in_indices = torch.cat([in_indices, index_next], dim=1)
            if eos_token_id is not None:
                finished = finished | (index_next.squeeze(1) == eos_token_id)
                if finished.all():
                    break
        return in_indices

