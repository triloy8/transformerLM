from einops import einsum, rearrange
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        # init W
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        # init W weights
        mean = 0.0
        std = 2/(in_features + out_features)
        a = mean - 3*std
        b = mean + 3*std
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
        return y
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # init
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        # init weights
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.weight[token_ids]
        return embeds

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model

        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1) + self.eps).unsqueeze(-1)

        x = (1/rms) * (x * self.weight)

        return x.to(in_dtype)
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        # init W
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        w3x = self.w3(x)

        silu = w1x * torch.sigmoid(w1x)
        glu = silu * w3x

        w2x = self.w2(glu)

        return w2x

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.device = device

        theta_i = theta ** (torch.arange(0, d_k, 2).float() / d_k)
        position = torch.arange(max_seq_len)

        phases = position.unsqueeze(1) / theta_i.unsqueeze(0)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        phases_combined = torch.stack([phases_cos, phases_sin], dim=-1).to(device=device)

        self.register_buffer("phases", phases_combined, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, '... (d_k p) -> ... d_k p', p=2)
        x1 = x[..., 0]
        x2 = x[..., 1]

        phases_cos = self.phases[..., 0][token_positions]
        phases_sin = self.phases[..., 1][token_positions]

        x_rotated = torch.stack([
            x1 * phases_cos - x2 * phases_sin,
            x1 * phases_sin + x2 * phases_cos
        ], dim=-1)

        return x_rotated.flatten(-2)
    
def softmax(x: torch.Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)

    return exp_x / sum_exp_x

def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        # one-hot at argmax
        argmax = probs.argmax(dim=-1)
        out = torch.zeros_like(probs)
        out.scatter_(-1, argmax.unsqueeze(-1), 1.0)
        return out
    if p >= 1:
        # renormalize defensively
        return probs / probs.sum(dim=-1, keepdim=True)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    cutoff = (cumulative >= p).float().argmax(dim=-1) + 1
    filtered = torch.zeros_like(probs)
    batch = probs.shape[0]
    for i in range(batch):
        k = cutoff[i].item()
        sel = sorted_indices[i, :k]
        sel_probs = sorted_probs[i, :k]
        filtered[i, sel] = sel_probs
        filtered[i] /= filtered[i].sum()
    return filtered

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    qk_score = einsum(Q, K, "batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m") / torch.sqrt(torch.tensor(Q.shape[-1]))
    masked_qk_score = qk_score.masked_fill(~mask, float('-inf'))
    softmax_masked_qk_score = softmax(masked_qk_score, dim=-1)
    attn = einsum(softmax_masked_qk_score, V, "batch_size ... n m, batch_size ... m d_k -> batch_size ... n d_k")

    return attn

class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads:int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len, device)
        
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool)))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        wqx = self.q_proj(x)
        wqx_rearr = rearrange(
            wqx,"... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads, d_k=self.d_k
        )
        wqx_rearr_rope = self.rope(wqx_rearr, token_positions)

        wkx = self.k_proj(x)
        wkx_rearr = rearrange(
            wkx,"... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads, d_k=self.d_k
        )
        wkx_rearr_rope = self.rope(wkx_rearr, token_positions)

        wvx = self.v_proj(x)
        wvx_rearr = rearrange(
            wvx,"... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads, d_v=self.d_v
        )

        seq_len = token_positions.shape[-1]
        attn = scaled_dot_product_attention(wqx_rearr_rope, wkx_rearr_rope, wvx_rearr, self.mask[:seq_len,:seq_len])

        attn_rearr = rearrange(
            attn,"... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
            num_heads=self.num_heads, d_v=self.d_v
        )

        attn_rearr_proj = self.output_proj(attn_rearr)
        
        return attn_rearr_proj

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
    
    @torch.no_grad() # in_indices: torch.Tensor
    def decode(self, in_indices: torch.Tensor, context_length: int, temperature=1.0, p=0.0, eos_token_id=None):
        batch_size = in_indices.shape[0]
        device = in_indices.device
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(context_length):
            context_indices = in_indices if in_indices.shape[1] <= context_length else in_indices[:, -context_length:]
            
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
