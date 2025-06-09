import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import (
    TransformerLayer,
    CosineScheduler,
    data2vecModule,
    EMAModule
)

class MLMModel(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.device = device
        self.n_layer = config.n_layer
        self.padding_idx = config.padding_idx
        self.N_idx = config.N_idx

        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=self.padding_idx)
        self.layer_norm_before = nn.LayerNorm(config.embed_dim)
        self.layer_norm_after = nn.LayerNorm(config.embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(config)
                for _ in range(config.n_layer)
            ]
        )
        self.classifer = nn.Linear(config.embed_dim, config.vocab_size)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.lr_scheduler = CosineScheduler(self.optimizer, config.warmup_iter, config.max_iter, config.learning_rate, config.min_lr)

        self.to(self.device)

    def loss_func(self, logits: torch.Tensor, token_seqs: torch.Tensor, mask_idxes: list):

        # "N"を予測に含めないようにするため，"N"を"<pad>"に置き換えてignore_indexで損失を求めないようにする
        target = token_seqs.clone()
        target[target == self.N_idx] = self.padding_idx		# (B,T)

        masked_indices = torch.zeros_like(token_seqs, dtype=torch.bool)
        for seq, mask_idx in enumerate(mask_idxes):
            masked_indices[seq][mask_idx] = True
        
        logits = logits[masked_indices]
        target = target[masked_indices]

        logits = logits.view(-1, logits.shape[-1])	# (B,T,C) => (BxT,C)
        target = target.view(-1)	# (B,T) => (BxT)

        criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        loss = criterion(logits, target)
        
        return loss
    
    def forward(self, tokens, padding_mask=None, attn_mask=None, repr_layers=[]):
        
        x = self.embed_tokens(tokens)		# (B, T) => (B, T, E)

        x = self.layer_norm_before(x)	# (B, T, E)

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        padding_mask_emb = padding_mask.unsqueeze(2).expand(-1, -1, x.shape[2])	# (B,T,E)
        for layer_idx, layer in enumerate(self.layers):
            x, _ = layer(x, padding_mask_emb=padding_mask_emb, attn_mask=attn_mask)		# (B, T, E)
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x

        x = self.layer_norm_after(x)		# (B, T, E)
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        # E => C: embed_dim => token_size
        x = self.classifer(x)		# (B, T, C)

        result = {"logits": x, "representations": hidden_representations}

        return result
    
    def _train(self, batch):
        x = batch["masked_token_seqs"].to(self.device)
        y = batch["token_seqs"].to(self.device)
        mask_idxes = batch["mask_idxes"]
        
        padding_mask = x.eq(self.padding_idx) # (B,T)

        B, T = x.shape
        Ls = batch["Ls"]
        attn_mask = torch.full((B, T, T), -1e6)
        for idx in range(B):
            attn_mask[idx, :Ls[idx], :Ls[idx]] = 0
        attn_mask = attn_mask.unsqueeze(1).to(self.device)

        result = self(x, padding_mask, attn_mask, repr_layers=[self.n_layer])
        logits = result["logits"]	# (B,T,C)

        result["loss"] = self.loss_func(logits, y, mask_idxes)

        repr = result["representations"][self.n_layer]
        result["repr_vars"] = torch.var(repr.view(-1, repr.shape[-1]), dim=0).mean()

        return result
    
    def calculate_repr(self, batch):
        y = batch["token_seqs"].to(self.device)
        
        padding_mask = y.eq(self.padding_idx) # (B,T)

        B, T = y.shape
        Ls = batch["Ls"]
        attn_mask = torch.full((B, T, T), -1e6)
        for idx in range(B):
            attn_mask[idx, :Ls[idx], :Ls[idx]] = 0
        attn_mask = attn_mask.unsqueeze(1).to(self.device)

        result = self(y, padding_mask, attn_mask, repr_layers=[self.n_layer])
        repr = result["representations"][self.n_layer]
        result["representation"] = repr
        repr = repr.view(-1, repr.shape[-1])
        result["repr_var"] = torch.var(repr, dim=0).mean()

        return result
    
    def _step(self):
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

class data2vecModel(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.device = device
        self.padding_idx = config.padding_idx
        self.ema_decay = config.ema_decay
        self.ema_end_decay = config.ema_end_decay
        self.ema_anneal_end_step = config.ema_anneal_end_step
        self.n_layer = config.n_layer
        self.k_layer = config.k_layer
        self.loss_beta = config.loss_beta
        self.num_updates = 0

        self.student_model = data2vecModule(config, device=self.device)
        self.ema = EMAModule(data2vecModule(config, device=self.device), self.ema_decay)

        curr_dim = config.embed_dim
        projs = []
        for i in range(config.n_head_layer - 1):
            next_dim = config.embed_dim * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, config.embed_dim))
        self.regression_head = nn.Sequential(*projs)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.lr_scheduler = CosineScheduler(self.optimizer, config.warmup_iter, config.max_iter, config.learning_rate, config.min_lr)

        self.to(self.device)
        

    def forward(self, masked_tokens, target_tokens=None, mask_idxes=None, padding_mask=None, attn_mask=None):
        x = self.student_model(masked_tokens, padding_mask=padding_mask, attn_mask=attn_mask, return_fc=False, repr_layers=[self.n_layer])[self.n_layer]
        student_var = torch.var(x.view(-1, x.shape[-1]), dim=0).mean()

        if target_tokens is None:
            result = {"representation": x, "repr_var": student_var}
            return result
        
        with torch.no_grad():
            # use EMA parameter as the teacher
            self.ema.model.eval()

            k_layers = list(range(self.n_layer - self.k_layer + 1, self.n_layer + 1))
            hidden_representations = self.ema.model(target_tokens, padding_mask=padding_mask, attn_mask=attn_mask, return_fc=True, repr_layers=k_layers)
            y = list(hidden_representations.values())
            assert len(y) == self.k_layer

            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]

            y = sum(y) / len(y)

            y = F.layer_norm(y.float(), y.shape[-1:])

        masked_indices = torch.zeros_like(masked_tokens, dtype=torch.bool)
        for seq, mask_idx in enumerate(mask_idxes):
            masked_indices[seq][mask_idx] = True

        teacher_var = torch.var(y.view(-1, y.shape[-1]), dim=0).mean()
        
        x = x[masked_indices]
        y = y[masked_indices]

        x = self.regression_head(x)

        sz = x.shape[-1]
        loss = F.smooth_l1_loss(x.float(), y.float(), reduction="none", beta=self.loss_beta).sum(dim=-1) / math.sqrt(sz)

        result = {"loss": loss.mean(), "ema_decay": self.ema.get_decay() * 1000, "repr_vars": (student_var, teacher_var)}

        return result

    
    def _train(self, batch):
        x = batch["masked_token_seqs"].to(self.device)
        y = batch["token_seqs"].to(self.device)
        mask_idxes = batch["mask_idxes"]
        
        padding_mask = x.eq(self.padding_idx) # (B,T)

        B, T = x.shape
        Ls = batch["Ls"]
        attn_mask = torch.full((B, T, T), -1e6)
        for idx in range(B):
            attn_mask[idx, :Ls[idx], :Ls[idx]] = 0
        attn_mask = attn_mask.unsqueeze(1).to(self.device)

        result = self(x, y, mask_idxes=mask_idxes, padding_mask=padding_mask, attn_mask=attn_mask)

        return result
    
    def calculate_repr(self, batch):
        y = batch["token_seqs"].to(self.device)
        
        padding_mask = y.eq(self.padding_idx) # (B,T)

        B, T = y.shape
        Ls = batch["Ls"]
        attn_mask = torch.full((B, T, T), -1e6)
        for idx in range(B):
            attn_mask[idx, :Ls[idx], :Ls[idx]] = 0
        attn_mask = attn_mask.unsqueeze(1).to(self.device)

        result = self(y, padding_mask=padding_mask, attn_mask=attn_mask)

        return result

    def set_num_updates(self):
        if self.training and self.ema is not None:
            if self.ema_decay != self.ema_end_decay:
                if self.num_updates >= self.ema_anneal_end_step:
                    decay = self.ema_end_decay
                else:
                    # 線形に更新割合を変化させる
                    r = self.ema_end_decay - self.ema_decay
                    pct_remaining = 1 - self.num_updates / self.ema_anneal_end_step
                    decay = self.ema_end_decay - r * pct_remaining
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.student_model)
        self.num_updates += 1
    
    def _step(self):
        self.optimizer.step()
        self.lr_scheduler.step()
        self.set_num_updates()
        self.optimizer.zero_grad()