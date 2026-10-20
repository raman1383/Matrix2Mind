from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True) # means that config can’t change mid-run
class AttentionConfig:
    num_q_heads: int
    num_kv_heads: int
    rope_theta: float


@dataclass(frozen=True)
class FFNConfig:
    use_gate: bool
    mlp_to_embed_expand_factor: int

    MoE_num_experts: int
    MoE_num_experts_per_tkn: int
    MoE_capacity_factor: float
    MoE_router_aux_loss_coef: float
    MoE_router_z_loss_coeff: float


@dataclass(frozen=True)
class ModelConfig:
    """Architecture only."""

    embed_dim: int
    vocab_size: int
    max_context_window: int
    num_layers: int = 0
    tied_embeddings: bool = False

    attention: AttentionConfig | None = None
    ffn: FFNConfig | None = None

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.attention.num_q_heads

    @property
    def num_groups(self) -> int:
        return self.attention.num_q_heads // self.attention.num_kv_heads

    @property
    def ffn_hidden_dim(self) -> int:
        return int(self.ffn.mlp_to_embed_expand_factor * self.embed_dim)

    def __post_init__(self):
        if self.attention is not None:
            if self.embed_dim % self.attention.num_q_heads != 0:
                raise ValueError("embed_dim must be divisible by num_q_heads")
            if self.attention.num_q_heads % self.attention.num_kv_heads != 0:
                raise ValueError("num_q_heads must be divisible by num_kv_heads")
            if self.head_dim % 2 != 0:
                raise ValueError("head_dim must be even for rotary embeddings")

        if self.ffn is not None and self.ffn.MoE_num_experts >= 1:
            if self.ffn.MoE_num_experts_per_tkn > self.ffn.MoE_num_experts:
                raise ValueError(
                    "MoE_num_experts must be >= MoE_num_experts_per_tkn"
                )


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    val_path: str


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    total_steps: int          # length of the WHOLE run; also the LR-schedule length
    batch_size: int
    lr: float

    run_root: str = "../runs"
    seed: int = 42

    # optimiser / schedule
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    amp: str = "none"         # "none" | "fp16" | "bf16"

    # eval / logging / checkpointing
    eval_every: int = 50
    eval_batches: int = 50    # SAME fixed subset for every point on the curve
    log_every: int = 10
    checkpoint_every: int = 1000
    keep_last_n: int = 3

    def __post_init__(self):
        object.__setattr__(self, "betas", tuple(self.betas))  # YAML gives a list
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be < total_steps")
        if self.amp not in ("none", "fp16", "bf16"):
            raise ValueError("amp must be 'none', 'fp16' or 'bf16'")

    @property
    def run_dir(self) -> Path:
        return Path(self.run_root) / self.run_name


def load_configs(path):
    """configs/*.YAML -> (ModelConfig, TrainConfig, DataConfig)."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    m = raw["model"]
    model = ModelConfig(
        **{k: v for k, v in m.items() if k not in ("attention", "ffn")},
        attention=AttentionConfig(**m["attention"]) if m.get("attention") else None,
        ffn=FFNConfig(**m["ffn"]) if m.get("ffn") else None,
    )
    return model, TrainConfig(**raw["train"]), DataConfig(**raw["data"])