import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class TransformerLayer(nn.Module):
    """
    Transformer layer block.
    正規化→attention→残差接続→正規化→全結合層→gelu→全結合層→残差接続
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.embed_dim = config.embed_dim
        self.ffn_embed_dim = config.ffn_embed_dim

        self.positional_embedding = RotaryPositionalEmbedding(config.max_length, config.embed_dim // self.n_head)
        self.layer_norm_before = nn.LayerNorm(config.embed_dim)
        self.layer_norm_after = nn.LayerNorm(config.embed_dim)
        self.c_attn = nn.Linear(config.embed_dim, 3*config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.fc1 = nn.Linear(config.embed_dim, config.ffn_embed_dim)
        self.fc2 = nn.Linear(config.ffn_embed_dim, config.embed_dim)

    def forward(self, x, padding_mask_emb=None, attn_mask=None, return_fc=False):
        B, T, E = x.shape

        residual = x
        x = self.layer_norm_before(x)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        q = q.view(B, T, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.positional_embedding(q, T)
        k = self.positional_embedding(k, T)

        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask
        )
        if attn_mask is not None:
            x = torch.nan_to_num(x)
        x = x.transpose(1, 2).contiguous().view(B, T, E)
        x = self.proj(x) * (~padding_mask_emb)
        x = residual + x

        residual = x
        x = self.layer_norm_after(x)
        if padding_mask_emb is not None:
            ffn_padding_mask_emb = torch.cat([padding_mask_emb for _ in range(self.ffn_embed_dim // self.embed_dim)], dim=-1)
            x = F.gelu(self.fc1(x) * (~ffn_padding_mask_emb))
            x = self.fc2(x) * (~padding_mask_emb)
        else:
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)

        fc_result = None
        if return_fc:
            fc_result = x
        
        x = residual + x

        return x, fc_result


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, max_length: int, head_size: int):
        super().__init__()
        self.head_size = head_size
        
        theta = 1.0 / (10000 ** (torch.arange(0, head_size, 2) / head_size))
        pos = torch.arange(max_length)
        x = torch.outer(pos, theta).repeat(1, 2)
        self.register_buffer("rope_cos", torch.cos(x), persistent=False)  # (max_length, head_size)
        self.register_buffer("rope_sin", torch.sin(x), persistent=False)  # (max_length, head_size)
    
    def forward(self, x, seq_length):
        rope_cos, rope_sin = self.rope_cos[:seq_length], self.rope_sin[:seq_length]
        x1 = x[..., : self.head_size // 2]
        x2 = x[..., self.head_size // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        roped = (x * rope_cos) + (rotated * rope_sin)
        return roped.type_as(x)  # (B, n_head, L, head_size)


class CosineScheduler(_LRScheduler):
    # 中田先生のやつのパクリ
    def __init__(self, optimizer, warmup_iter=10000, max_iter=200000, learning_rate=1e-5, min_lr=1e-6, last_iter=-1):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_iter)
    
    def get_lr(self):
        last_iter = self.last_epoch
        if last_iter < self.warmup_iter:
            return [base_lr * last_iter / self.warmup_iter for base_lr in self.base_lrs]
        if last_iter > self.max_iter:
            return [base_lr * self.min_lr / self.learning_rate for base_lr in self.base_lrs]
        decay_ratio = (last_iter - self.warmup_iter) / (self.max_iter - self.warmup_iter)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return [self.min_lr + coeff * (base_lr - self.min_lr) for base_lr in self.base_lrs]


class EMAModule:
    """Exponential Moving Average of Fairseq Models"""

    def __init__(
        self,
        model,
        ema_decay=0.9999,
        device=None,
        skip_keys=None,
    ):
        # modelは今のteacher modelを指す
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.model.to(device)
        self.decay = ema_decay
        self.skip_keys = skip_keys or set()	# studentのparameterのうち，追跡をしない部分．基本Noneで使うつもり

        
    def set_decay(self, decay, weight_decay=None):
        self.decay = decay
        if weight_decay is not None:
            self.weight_decay = weight_decay

    def get_decay(self):
        return self.decay
    
    def _step_internal(self, new_model):
        """One update of the EMA model based on new model (student model) weights"""
        decay = self.decay

        ema_state_dict = {}
        ema_params = self.model.state_dict()

        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue

            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = (param.float().clone() if param.ndim == 1 else copy.deepcopy(param))
                ema_params[key] = ema_param

            if param.shape != ema_param.shape:
                raise ValueError("incompatible tensor shapes between model param and ema param"+ "{} vs. {}".format(param.shape, ema_param.shape))

            if "version" in key:
                # Do not decay a model.version pytorch param
                continue

            if key in self.skip_keys or not param.requires_grad:
                ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                ema_param.mul_(decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1-decay)

            ema_state_dict[key] = ema_param

        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        self.model.load_state_dict(ema_state_dict, strict=False)

    @torch.no_grad()
    def step(self, new_model):
        self._step_internal(new_model)

    def reverse(self, model):
        """
        Load the model parameters from EMA model.
        Useful for inference or fine-tuning from the EMA model.
        """
        d = self.model.state_dict()
        if "_ema" in d:
            del d["_ema"]

        model.load_state_dict(d, strict=False)
        return model


# teacherとstudentが使う共通のアーキテクチャの実装
class data2vecModule(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.device = device
        self.padding_idx = config.padding_idx

        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=self.padding_idx)
        self.layer_norm_before = nn.LayerNorm(config.embed_dim)
        self.layer_norm_after = nn.LayerNorm(config.embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(config)
                for _ in range(config.n_layer)
            ]
        )

        self.to(self.device)
    
    def forward(self, tokens, padding_mask=None, attn_mask=None, return_fc=False, repr_layers=[]):
        
        x = self.embed_tokens(tokens)		# (B, T) => (B, T, E)

        x = self.layer_norm_before(x)	# (B, T, E)

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        padding_mask_emb = padding_mask.unsqueeze(2).expand(-1, -1, x.shape[2])	# (B,T,E)
        for layer_idx, layer in enumerate(self.layers):
            x, fc_result = layer(x, padding_mask_emb=padding_mask_emb, attn_mask=attn_mask, return_fc=return_fc)		# (B, T, E)
            if (layer_idx + 1) in repr_layers:
                if return_fc:
                    hidden_representations[layer_idx + 1] = fc_result
                else:
                    hidden_representations[layer_idx + 1] = x
        
        if return_fc:
            return hidden_representations

        x = self.layer_norm_after(x)		# (B, T, E)
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        return hidden_representations

