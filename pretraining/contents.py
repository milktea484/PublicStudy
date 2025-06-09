from dataclasses import dataclass

rnaseq_tokens = ["A", "C", "G", "U", "N"]

appended_tokens = ["A", "C", "G", "U", "N", "<mask>", "<pad>"]


@dataclass
class ModelConfig:
    max_iter: int = 50000
    warmup_iter: int = 5000
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    vocab_size: int = len(appended_tokens)
    max_length: int = 512
    embed_dim: int = 128
    ffn_embed_dim: int = 256
    n_layer: int = 6
    n_head: int = 8
    learning_rate: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    eval_interval: int = 1000
    eval_iter: int = 100
    padding_idx: int = appended_tokens.index("<pad>")
    N_idx: int = appended_tokens.index("N")
    
@dataclass
class data2vecConfig(ModelConfig):
    k_layer: int = 3
    ema_decay: float = 0.999
    ema_end_decay: float = 0.9999
    ema_anneal_end_step: int = 20000
    loss_beta: float = 4.0
    n_head_layer: int = 1
