import torch
from pathlib import Path
from typing import Optional
from collections import deque
import torch.nn.functional as F
from dataclasses import dataclass, field

from kv_cache import KV_cache
from model_config import ModelConfig
from tokenizerModule import ID_mapper

def load_model_weights(model: torch.nn.Module, save_path: str, device) -> torch.nn.Module:
    """Load a checkpoint into `model` and return it in eval mode."""
    path = Path(save_path)
 
    if not path.is_file():
        raise FileNotFoundError(f"[ERR] Weight file not found: {path.resolve()}")
 
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"[ERR] Could not read checkpoint {path}: {e}") from e
 
    # Raises RuntimeError listing missing / unexpected keys or shape mismatches.
    # (Checkpoints from before the fused-QKV / tied-embedding change won't load: retrain.)
    model.load_state_dict(state_dict)
 
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Loaded weights from {path} ({n_params:,} parameters)")
    return model

# ----------------------------------------------------------------------


def sample_next_token_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Sample one next token from logits.

    Input:
        logits: [B, Vocab_Size]

    Output:
        next_token: [B, 1]
    """

    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [B, V], got {tuple(logits.shape)}"
        )

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if temperature == 0.0:
        # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive or None")
        top_k = min(top_k, logits.shape[-1])

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1] or None")

    # 1. Temperature scaling
    logits = logits / temperature

    # 2. Top-k filtering
    if top_k is not None:
        kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))


    # 3. Top-p filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep the first token that crosses the threshold.
        sorted_indices_to_remove[..., 1:] = (
            sorted_indices_to_remove[..., :-1].clone()
        )
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )


        logits = logits.masked_fill(
            indices_to_remove,
            float("-inf"),
        )

    # 4. Convert to probabilities and sample
    probabilities = F.softmax(logits, dim=-1)

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )

# ----------------------------------------------------------------------
# NAIVE BASELINE: no cache, recompute everything every step.
#
# This is the "obviously correct" reference the cached engine is tested against,
# and the slow side of the slow-vs-fast demo. It takes an ALREADY LOADED model
# (use load_model_weights first); it never touches the disk.
# ----------------------------------------------------------------------

def _sliding_window_mask(seq_len: int, window: int, device) -> torch.Tensor:
    """Bool [1, 1, T, T]: query p may attend key a iff p - window < a <= p."""
    i = torch.arange(seq_len, device=device)
    allowed = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)
    return allowed[None, None]


@torch.inference_mode()
def batched_simple_inference_ids(
    model: torch.nn.Module,
    configs: ModelConfig,
    prompt_ids: list[list[int]],
    num_gen_steps: int,
    device,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
) -> list[list[int]]:
    """
    Generate `num_gen_steps` tokens for every prompt by re-running the WHOLE growing
    sequence through the model at every step (quadratic total work).

    Once a sequence is longer than the context window, attention uses the same
    sliding-window rule as the KV cache (query p sees keys in (p - W, p], absolute
    RoPE positions), so at temperature 0 this matches advanced_inference exactly,
    for any length.

    All prompts must have the same number of tokens (they are stacked into one tensor).

    Returns only the generated ids: a list of `num_gen_steps` ints per prompt.
    """
    if num_gen_steps < 0:
        raise ValueError("num_gen_steps must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must contain at least one prompt")

    lengths = {len(p) for p in prompt_ids}
    if 0 in lengths:
        raise ValueError("prompts must contain at least one token")
    if len(lengths) != 1:
        raise ValueError(
            f"All prompts must have the same token length, got lengths: "
            f"{[len(p) for p in prompt_ids]}"
        )
    prompt_len = lengths.pop()

    model.eval()
    window = configs.max_context_window

    # [B, prompt_len]; grows by one column per step
    sequences = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    for _ in range(num_gen_steps):
        seq_len = sequences.shape[1]

        # T <= window: plain causal attention (fast path, mask=None).
        # T >  window: explicit sliding-window mask.
        mask = _sliding_window_mask(seq_len, window, sequences.device) if seq_len > window else None

        logits = model(sequences, attn_mask=mask)        # [B, T, V]

        next_token = sample_next_token_from_logits(      # [B, V] -> [B, 1]
            logits[:, -1, :], temperature, top_k, top_p
        )
        sequences = torch.cat([sequences, next_token], dim=-1)

    return sequences[:, prompt_len:].tolist()


def batched_simple_inference(
    model: torch.nn.Module,
    configs: ModelConfig,
    num_gen_steps: int,
    prompt_strs: list[str],
    device,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
) -> list[str]:
    """
    String version of batched_simple_inference_ids.
    Each output string is its prompt plus `num_gen_steps` generated tokens.
    The model must already hold trained weights (see load_model_weights).
    """
    prompt_ids = [ID_mapper.encode(p) for p in prompt_strs]
    generated = batched_simple_inference_ids(
        model, configs, prompt_ids, num_gen_steps, device,
        temperature=temperature, top_k=top_k, top_p=top_p,
    )
    return [ID_mapper.decode(p + g) for p, g in zip(prompt_ids, generated)]


# ----------------------------------------------------------------------



@dataclass
class _Request:
    index: int
    prompt_ids: list # FULL prompt, never truncated
    max_new_tokens: int
    n_prefilled: int = 0 # prompt tokens already in the cache
    generated: list = field(default_factory=list)
    finish_reason: str = ""


@dataclass
class TokenCompletion:
    generated: list      # generated ids (the EOS token is not included)
    finish_reason: str   # "eos" | "length"


@dataclass
class Completion:
    text: str            # prompt + generated
    completion: str      # generated only
    finish_reason: str   # "eos" | "length"



@torch.inference_mode()
def advanced_inference_ids(

    model,
    configs: ModelConfig,
    prompt_ids: list[list[int]],
    
    device,
    max_new_tokens: int = 64,

    max_KV_lanes: int = 4,               # max sequences in flight (cache lanes)
    chunk_size: int = 32,             # prompt tokens prefilled per iteration


    temperature: float = 0.6,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,

) -> list[TokenCompletion]:

    
    """

    Iteration-level scheduling with chunked prefill and a ring-buffer KV cache.
    Prompts and generations may be arbitrarily long; the cache stays at
    max_context_window tokens per lane. Returns one TokenCompletion per prompt,
    in input order. Works on token ids, so it needs no tokenizer.

    max_new_tokens=0 returns immediately with empty generations and never runs the model.
    
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_KV_lanes < 1:
        raise ValueError("max_KV_lanes must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    model.eval()

    waiting = deque()
    finished:   dict[int, _Request] = {}     # input index -> done

    for i, ids in enumerate(prompt_ids):
        if not ids:
            raise ValueError(f"Prompt {i} encodes to zero tokens")
        req = _Request(i, list(ids), max_new_tokens)
        if max_new_tokens == 0:
            req.finish_reason = "length"     # nothing to generate: no lane, no compute
            finished[i] = req
        else:
            waiting.append(req)


    if waiting:
        num_lanes = min(max_KV_lanes, len(waiting))

        cache = KV_cache(
            configs, 
            num_lanes=num_lanes, 
            device=device,
            dtype=next(model.parameters()).dtype
        )

        free_lanes = list(range(num_lanes))

        prefilling: dict[int, _Request] = {}     # lane -> prompt not fully consumed yet
        running:    dict[int, _Request] = {}     # lane -> generating


        def sample(logits):                      
            # [B,V] -> [B,1]; 
            return sample_next_token_from_logits(logits, temperature, top_k, top_p)


        def accept(lane, req, tok):
            req.generated.append(tok)
            
            if eos_token_id is not None and tok == eos_token_id:
                req.finish_reason = "eos"
            
            elif len(req.generated) >= req.max_new_tokens:
                req.finish_reason = "length"
            
            if req.finish_reason:                # retire: free the lane immediately
                cache.free(lane)
                free_lanes.append(lane)
                del running[lane]
                finished[req.index] = req


        while waiting or prefilling or running:

            # 1) ADMIT: claim a lane, no compute yet
            while waiting and free_lanes:
                prefilling[free_lanes.pop()] = waiting.popleft()

            # 2) PREFILL: one chunk of one request per iteration, so decode never stalls long
            if prefilling:
                lane, req = next(iter(prefilling.items()))
                lo = req.n_prefilled
                hi = min(lo + chunk_size, len(req.prompt_ids))
                ids    = torch.tensor([req.prompt_ids[lo:hi]], device=device)
                # lane ids stay on the CPU: the cache keeps a host-side length mirror,
                # so building the mask needs no GPU -> CPU sync
                logits = model(ids, cache=cache, lane_ids=torch.tensor([lane]))
                req.n_prefilled = hi
                if hi == len(req.prompt_ids):    # prompt done -> sample first token
                    del prefilling[lane]
                    running[lane] = req
                    accept(lane, req, sample(logits[:, -1, :]).item())


            # 3) DECODE: one batched step over every generating request
            if running:
                lane     = list(running)
                lane_ids = torch.tensor(lane)
                last     = torch.tensor([[running[s].generated[-1]] for s in lane], device=device)
                logits   = model(last, cache=cache, lane_ids=lane_ids)
                for s, t in zip(lane, sample(logits[:, -1, :])[:, 0].tolist()):
                    accept(s, running[s], t)


    results = []
    for i in range(len(prompt_ids)):
        r   = finished[i]
        gen = r.generated[:-1] if r.finish_reason == "eos" else r.generated
        results.append(TokenCompletion(generated=gen, finish_reason=r.finish_reason))
    return results


def advanced_inference(

    model,
    configs: ModelConfig,
    prompt_strs: list[str],
    
    device,
    max_new_tokens: int = 64,

    max_KV_lanes: int = 4,
    chunk_size: int = 32,

    temperature: float = 0.6,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,

) -> list[Completion]:
    """String version of advanced_inference_ids. Returns one Completion per prompt, in input order."""

    prompt_ids = [ID_mapper.encode(p) for p in prompt_strs]

    outs = advanced_inference_ids(
        model, configs, prompt_ids, device,
        max_new_tokens=max_new_tokens, max_KV_lanes=max_KV_lanes, chunk_size=chunk_size,
        temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
    )

    return [
        Completion(
            text=ID_mapper.decode(p + o.generated),
            completion=ID_mapper.decode(o.generated),
            finish_reason=o.finish_reason,
        )
        for p, o in zip(prompt_ids, outs)
    ]