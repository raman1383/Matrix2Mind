import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from tensor_tools import init_custom_weight
from model_config import ModelConfig


# FFN helper functions 
def _zero_aux(x: torch.Tensor) -> torch.Tensor:
    return x.new_zeros((), dtype=torch.float32)

def _ffn_activation(h: torch.Tensor, gated: bool) -> torch.Tensor:
    """h: [..., 2*H] ([gate | up]) if gated else [..., H]  ->  [..., H]"""
    if gated:
        gate, up = h.chunk(2, dim=-1)
        return F.silu(gate) * up
    return F.silu(h)




class MLP_SubBlock(nn.Module):
    """Position-wise FFN: D -> H -> D with SiLU. Each token is processed independently."""

    def __init__(self, cfg:ModelConfig):
        super().__init__()

        self.num_layers = cfg.num_layers

        self.W_up   = nn.Parameter(torch.empty(
            cfg.embed_dim, cfg.ffn_hidden_dim
            )
        )
        
        self.W_down = nn.Parameter(torch.empty(
            cfg.ffn_hidden_dim, cfg.embed_dim
            )
        )

        self._init_weights()


    def _init_weights(self):
        init_custom_weight(self.W_up,   is_residual_output=False)
        
        init_custom_weight(self.W_down, is_residual_output=True , 
                                        num_layers=self.num_layers)

  
    def forward(self, x:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        out = _ffn_activation(x @ self.W_up, gated=False) @ self.W_down
        return out, _zero_aux(x)
        


class GLU_SubBlock(nn.Module):
    """Gated FFN (SwiGLU). Params: W_gate_up [D, 2H] (= [gate | up]) and W_down [H, D]."""

    def __init__(self, cfg:ModelConfig):
        super().__init__()

        self.num_layers = cfg.num_layers
        self.hidden_dim = cfg.ffn_hidden_dim

        self.W_gate_up  = nn.Parameter(torch.empty(
            cfg.embed_dim, 2 * self.hidden_dim
            )
        )

        
        self.W_down = nn.Parameter(torch.empty(
            self.hidden_dim, cfg.embed_dim
            )
        )


        self._init_weights()


    def _init_weights(self):
        H = self.hidden_dim

        # initialise the two halves separately so each sees a [D, H] matrix
        init_custom_weight(self.W_gate_up[:, :H],   is_residual_output=False) # gate
        init_custom_weight(self.W_gate_up[:, H:],   is_residual_output=False) # up

        init_custom_weight(self.W_down, is_residual_output=True, 
                                        num_layers=self.num_layers)


    def forward(self, x):
        out = _ffn_activation(x @ self.W_gate_up, gated=True) @ self.W_down
        return out, _zero_aux(x)
    





class Mixture_of_Experts_SubBlock(nn.Module):

    """
    Sparse Mixture-of-Experts FFN: every token is processed by only `top_k` of the
    `num_experts` expert FFNs, chosen by a learned router. Parameters grow with
    num_experts, compute per token only with top_k.
 
    Implementation = "capacity-based dense dispatch" (GShard / Switch Transformer style):
 
      1. ROUTE      router logits -> softmax -> top-k experts + gate weights per token.
      2. ASSIGN     give each (token, expert) pair a slot  0..C-1  inside that expert's buffer
                    (C = expert capacity). Pairs that don't fit are dropped.
      3. DISPATCH   scatter tokens into ONE dense buffer  [num_experts, C, embed_dim]   (index_add, no Python loop)
      4. COMPUTE    all experts at once with two batched matmuls (torch.bmm) over stacked
                    3-D weights  [num_experts, embed_dim, H] / [num_experts, H, embed_dim]   (no per-expert Python loop)
      5. COMBINE    gather every pair's output back, weight by its gate, sum over the k slots.
 
    Why this design
      * No data-dependent shapes and no `.item()` / `.tolist()` host syncs: shapes depend
        only on (B*T, num_experts, k, capacity_factor) -> torch.compile / CUDA-graph friendly.
      * Two big bmm calls instead of num_experts small matmuls -> far fewer kernel launches.
      * FLOPs ~ capacity_factor * (dense FFN FLOPs) * top_k, independent of num_experts.
 
    Capacity / token dropping
      C = ceil(capacity_factor * N * top_k / num_experts)   with N = B*T tokens in this call.
      * TRAINING with capacity_factor set: overflowing pairs are dropped (their expert
        contributes 0; the residual connection still carries the token) and priority goes
        to 1st-choice assignments over 2nd-choice, then to earlier tokens.
      
      * EVAL, or capacity_factor=None: C = N, which can never overflow (one token picks each
        expert at most once) -> exact, drop-free output. It wastes padding FLOPs, but at
        inference N is small (a few lanes x chunk), so that is the right trade-off, and it
        keeps results independent of what else is in the batch.
 
    Load balancing (Switch Transformer): aux = num_experts * sum_e f_e * P_e
        f_e = fraction of (token, slot) assignments routed to expert e   (not differentiable)
        P_e = mean router probability of expert e                        (differentiable)
      Minimised (=1.0) when both are uniform. Optional router z-loss (ST-MoE) penalises large
      router logits for numerical stability.
 
    Optional FFNConfig fields:
        capacity_factor        (default 1.25, None -> never drop)
        router_aux_loss_coef   (default 0.01)
        router_z_loss_coef     (default 0.0)
    """


    def __init__(self, cfg:ModelConfig):
        super().__init__()

        self.embed_dim   = cfg.embed_dim
        self.hidden_dim  = cfg.ffn_hidden_dim # per-expert hidden size
        self.use_gate    = cfg.ffn.use_gate
        self.num_layers  = cfg.num_layers
        self.num_experts = cfg.ffn.MoE_num_experts
        self.top_k       = cfg.ffn.MoE_num_experts_per_tkn

        self.capacity_factor = cfg.ffn.MoE_capacity_factor
        self.aux_loss_coef   = cfg.ffn.MoE_router_aux_loss_coef
        self.z_loss_coef    = cfg.ffn.MoE_router_z_loss_coeff


        if self.num_experts < 2:
            raise ValueError("MoE needs num_experts >= 2 (use GLU/MLP for a single expert)")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("need 1 <= num_experts_per_tkn <= num_experts")
 

        # gated: [gate | up] fused, as in GLU
        up_dim = 2 * self.hidden_dim if self.use_gate \
                                    else self.hidden_dim          


        # Stacked expert weights: one 3-D tensor per matrix instead of ModuleList of num_experts modules
        # -> a single batched-matrix-multiply covers all experts.

        
        self.router = nn.Parameter(torch.empty(self.embed_dim,   self.num_experts))
        self.W_in   = nn.Parameter(torch.empty(self.num_experts, self.embed_dim,  up_dim))   # [gate|up] if gated else up
        self.W_down = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.embed_dim))

        self._init_weights()


    def _init_weights(self):
        init_custom_weight(self.router, is_residual_output=False)
        
        # Experts must be initialised one by one: init_custom_weight expects 2-D matrices
        # (fan-in / fan-out), and a 3-D init would also give every expert the same scale
        # computed from the wrong dimensions.
        H = self.hidden_dim
        for e in range(self.num_experts):
            if self.use_gate:
                init_custom_weight(self.W_in[e, :, :H], is_residual_output=False)   # gate
                init_custom_weight(self.W_in[e, :, H:], is_residual_output=False)   # up
            else:
                init_custom_weight(self.W_in[e], is_residual_output=False)
           
            init_custom_weight(self.W_down[e], is_residual_output=True,
                                               num_layers=self.num_layers)


    def _capacity(self, num_tokens: int) -> int:
        if self.training and self.capacity_factor is not None:
            c = math.ceil(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
            return min(num_tokens, max(1, c))
        return num_tokens # drop-free


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = x.shape
        BxS = batch_size * seq_len
        x_flat   =  x.reshape(BxS, embed_dim)


        # ---- 1. ROUTE (in fp32: bf16 logits make near-ties / unstable routing) ------------
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = x_flat.float() @ self.router.float() # [BxS, embed_dim]   -> [BxS, num_experts]
            probs  = logits.softmax(dim=-1)               # [BxS, num_experts]   -> [BxS, num_experts]
            
        gate_w, expert_idx = probs.topk(self.top_k, dim=-1)    # [BxS, top_k] each

        if self.top_k > 1:
            # renormalise so the K gates sum to 1 (== softmax over the top-k logits, Mixtral).
            # NOT done for K=1: the gate would be the constant 1.0 and the router would get no
            # gradient from the language-modelling loss. Switch keeps the raw probability.
            gate_w = gate_w / gate_w.sum(dim=-1, keepdim=True)



        # ---- 2. ASSIGN each (token, slot) pair a position inside its expert's buffer -------
        
        # Flatten SLOT-MAJOR: [all tokens' 1st choice, then all tokens' 2nd choice, ...]
        # so that on overflow the 1st choices win the capacity race.   a = slot * N + token
        flat_expert = expert_idx.t().reshape(-1) # [top_k * BxS]
        onehot      = F.one_hot(flat_expert, self.num_experts)                            # [K*N, self.num_experts]
        
        # position = how many earlier pairs already went to the same expert
        pos = onehot.cumsum(dim=0).gather(1, flat_expert[:, None]).squeeze(1) - 1   # [K*N]
 
        C     = self._capacity(BxS)
        keep  = pos < C
        trash = self.num_experts * C                       # extra row that absorbs dropped pairs
        dest  = torch.where(keep, flat_expert * C + pos, torch.full_like(pos, trash))  # [K*N]



        # ---- 3. DISPATCH: scatter token copies into a flat [E*C (+1 trash), D] buffer -------

        # Kept pairs have unique destinations, so index_add == plain assignment; it is also
        # differentiable w.r.t. the tokens. Empty slots stay zero (and give zero output).
        x_rep = x_flat.repeat(self.top_k, 1)                                        # [K*N, D], slot-major
        buf   = x_flat.new_zeros(self.num_experts * C + 1, embed_dim).index_add(0, dest, x_rep)
        x_exp = buf[:-1].view(self.num_experts, C, embed_dim)                                     # [E, C, D]


        # ---- 4. COMPUTE all experts at once ------------------------------------------------

        h     = torch.bmm(x_exp, self.W_in)                                # [E, C, up_dim]
        h     = _ffn_activation(h, self.use_gate)                          # [E, C, H]
        y_exp = torch.bmm(h, self.W_down)                                  # [E, C, D]



        # ---- 5. COMBINE: gather back, weight by gate, sum the K slots ----------------------

        y_flat = F.pad(y_exp.reshape(self.num_experts * C, embed_dim), (0, 0, 0, 1))              # zero "trash" row
        y_rep  = y_flat[dest]                                              # [K*N, D]; dropped -> 0
        w      = gate_w.t().reshape(-1, 1).to(y_rep.dtype)                 # [K*N, 1], slot-major
        y      = (y_rep * w).view(self.top_k, BxS, embed_dim).sum(dim=0)                      # [N, D]



        # ---- auxiliary losses (training only; skipped at inference) ------------------------

        if self.training:
            f   = onehot.sum(dim=0).float() / (BxS * self.top_k)       # [E] share of assignments (no grad)
            P   = probs.mean(dim=0)                         # [E] mean router prob   (has grad)
            aux = self.aux_loss_coef * self.num_experts * (f * P).sum()
            if self.z_loss_coef > 0:
                aux = aux + self.z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()
        else:
            aux = _zero_aux(x)
 
        return y.view(batch_size, seq_len, embed_dim), aux


def build_ffn_sub_block(cfg: ModelConfig) -> nn.Module:
    ffn = cfg.ffn
    if ffn.MoE_num_experts < 1:
        raise ValueError("num_experts must be >= 1")
    if ffn.MoE_num_experts > 1:
        return Mixture_of_Experts_SubBlock(cfg)      # honours use_gate internally
    
    return GLU_SubBlock(cfg) if ffn.use_gate \
                             else MLP_SubBlock(cfg)
