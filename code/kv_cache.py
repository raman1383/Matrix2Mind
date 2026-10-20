import torch
from model_config import ModelConfig


def _to_host(lane_ids) -> list[int]:
    """Lane ids as plain Python ints. Free for lists / CPU tensors (a sync for CUDA tensors)."""
    if isinstance(lane_ids, torch.Tensor):
        return lane_ids.tolist()
    return [int(l) for l in lane_ids]


class KV_cache:
    """
    Per-lane ring buffer.
      keys/values: [num_layers, num_lanes, num_kv_heads, cache_window, head_dim]
      lengths:     [num_lanes] -> the number of sequences in flight, and finished ones hand their lane to waiting ones

    Token with absolute position a lives at physical index absolute_pos % cache_window.

    The layout of one forward pass depends only on its shape (chosen in prepare(), used in update()):

      DECODE  (q_len == 1: one new token per lane, the hot loop)
        WRITE the new token into its ring slot first, then attend over the ring IN PLACE.
        Safe for exactly one new token: the slot it overwrites holds position p - W, which the
        sliding window (p - W < a <= p) excludes anyway. The mask is built over PHYSICAL slots,
        so keys are never reordered or concatenated (attention is permutation-invariant over
        keys, and RoPE was applied before caching). If the lanes of the batch are consecutive
        and ascending (e.g. [0, 1, 2]) the keys handed to attention are plain VIEWS of the
        cache: zero bytes copied per layer per step. Otherwise one gather copies only the
        slots in use.

      PREFILL CHUNK  (q_len > 1)
        READ the old ring first, append the new chunk, THEN write the last min(T, W) chunk
        tokens. This order is required for chunks: a chunk's early queries still need ring
        entries that the chunk's later tokens overwrite. The only copy is the concatenation
        (the old ring is read through a view when the lanes are consecutive, which a single
        prefilling request always is). It runs once per prompt chunk, not once per token.

    Host mirror
      `lengths` (on the device) is what the mask math uses. `_host_len` is a plain Python
      list that is always equal to it. prepare() needs one number on the CPU (how many
      slots to read), and reading it from the device tensor would force a
      GPU -> CPU sync on every forward pass. The mirror makes that read free.
      For the mirror to stay free, pass lane_ids as a Python list or a CPU tensor.
      A CUDA tensor still works, it just costs one sync (`.tolist()`) per forward.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_lanes: int,
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float32,

    ):
        self.cache_window   = config.max_context_window

        shape = (config.num_layers,
                 num_lanes,
                 config.attention.num_kv_heads,
                 self.cache_window,
                 config.head_dim
                )

        self.keys    = torch.zeros(shape, device=device, dtype=dtype)
        self.values  = torch.zeros(shape, device=device, dtype=dtype)
        self.device  = self.keys.device

        self.lengths   = torch.zeros(num_lanes, dtype=torch.long, device=self.device) # ABSOLUTE count(no wrap-around)
        self._host_len = [0] * num_lanes                                              # CPU mirror of `lengths`

        # lane ids of the forward pass in flight, converted ONCE in prepare()
        # and reused by update() (every layer) and commit()
        self._lane_src  = None   # the object the caller passed (identity check)
        self._lane_dev  = None   # same lanes as a device tensor
        self._lane_host = None   # same lanes as a list[int]

        # plan of the forward pass in flight, computed ONCE in prepare() and reused by update() in every layer
        self._q_len      = 0       # T
        self._decode     = False   # T == 1 -> write first, attend over the ring in place
        self._m          = 0       # chunk tokens written to the ring: min(T, W)
        self._phys       = None    # [B, m] physical ring slots those tokens are written to
        self._n_read     = 0       # ring slots attention reads (decode: after the write; chunk: before it)
        self._lane_slice = None    # slice(l0, l0+B) if the lanes are consecutive ascending, else None


    # ------------------------------------------------------------------
    def _convert(self, lane_ids):
        host = _to_host(lane_ids)
        if isinstance(lane_ids, torch.Tensor):
            dev = lane_ids.to(self.device, dtype=torch.long)
        else:
            dev = torch.tensor(host, dtype=torch.long, device=self.device)
        return dev, host


    def _resolve(self, lane_ids):
        """Reuse prepare()'s conversion when the caller passes the same object again."""
        if self._lane_src is not None and lane_ids is self._lane_src:
            return self._lane_dev, self._lane_host
        return self._convert(lane_ids)


    def free(self, lane: int):
        # no need to zero memory; the mask hides stale data
        self.lengths[lane] = 0
        self._host_len[lane] = 0


    # ONCE per forward, after ALL layers
    def commit(self, lane_ids, q_len):
        # absolute tokens seen per lane, never wraps.
        # RoPE and the mask use absolute positions (lengths + arange(T))
        dev, host = self._resolve(lane_ids)
        self.lengths[dev] += q_len
        for lane in host:
            self._host_len[lane] += q_len


    # runs once before the layers and returns position_ids and a bool mask
    #   decode (T == 1):  mask [B, 1, 1, n_read]      over the ring's physical slots
    #   chunk  (T  > 1):  mask [B, 1, T, n_read + T]  over keys [old ring | new chunk]
    def prepare(self, lane_ids, q_len):
        W = self.cache_window

        dev_ids, host_ids = self._convert(lane_ids)
        self._lane_src, self._lane_dev, self._lane_host = lane_ids, dev_ids, host_ids

        start = self.lengths[dev_ids]   # [B] ABSOLUTE tokens seen so far
        dev = start.device

        # absolute position of every token in this chunk. Drives RoPE. Never wraps.
        pos = start[:, None] + torch.arange(q_len, device=dev)[None, :]      # [B, T]

        # Which ring slots does this forward write? The last m = min(T, W) chunk tokens.
        # Computed once here; update() reuses it in every layer instead of re-deriving it.
        m = min(q_len, W)
        self._q_len = q_len
        self._m     = m
        self._phys  = pos[:, q_len - m:] % W                                 # [B, m]

        # Consecutive ascending lanes -> the ring can be read through a view (no gather).
        l0, B = host_ids[0], len(host_ids)
        self._lane_slice = slice(l0, l0 + B) if host_ids == list(range(l0, l0 + B)) else None

        # Longest history in the batch, from the host mirror: no GPU -> CPU sync.
        max_len = max(self._host_len[l] for l in host_ids)

        self._decode = (q_len == 1)
        if self._decode:
            # The new token is written BEFORE attention reads, so its slot is part of the read.
            # Slots 0..n_read-1 are the only ones any lane in this batch has ever written.
            self._n_read = min(max_len + 1, W)

            # pos is [B, 1] = the absolute position of the token being written = the newest one.
            # Absolute position held by physical slot j once that token is in the ring:
            #     a_j = p - (p - j) % W        (negative => this slot was never written by this lane)
            # Every written slot satisfies p - W < a_j <= p, so only "was it ever written" needs masking.
            j = torch.arange(self._n_read, device=dev)
            a = pos - (pos - j[None, :]) % W                                 # [B, n_read]
            return pos, (a >= 0)[:, None, None, :]                           # [B, 1, 1, n_read]

        # ---- prefill chunk ---------------------------------------------------------------
        # what the ring currently holds (state BEFORE this chunk is written).
        n_old = min(max_len, W)
        self._n_read = n_old
        j     = torch.arange(n_old, device=dev)
        last  = (start - 1)[:, None]                                 # newest old position, -1 if empty

        a_old = last - (last - j[None, :]) % W                       # abs position at each physical idx
                                                                    # (negative => never written)
        # keys seen by attention = [old ring | new chunk]
        a_all = torch.cat([a_old, pos], dim=1)                       # [B, n_old + T]

        a, p = a_all[:, None, :], pos[:, :, None]
        mask = (a >= 0) & (a <= p) & (a > p - W)                     # exists, causal, in window

        return pos, mask[:, None]                                    # [B, 1, T, n_old + T]


    def update(self, layer_idx, lane_ids, k, v):
        if self._phys is None:
            raise RuntimeError("KV_cache.update() was called before prepare()")

        B, H, T, D = k.shape
        if T != self._q_len:
            raise RuntimeError(
                f"update() got T={T} but prepare() was called with q_len={self._q_len}")

        lane_ids, _ = self._resolve(lane_ids)
        K, V = self.keys[layer_idx], self.values[layer_idx]
        k, v = k.to(K.dtype), v.to(K.dtype)

        phys = self._phys                                       # [B, m], from prepare()
        n    = self._n_read
        sl   = self._lane_slice

        if self._decode:
            # 1) WRITE the single new token. Its slot held position p - W, already outside the window.
            slot = phys[:, 0]                                   # [B]
            K[lane_ids, :, slot] = k[:, :, 0]                   # [B, H, D]
            V[lane_ids, :, slot] = v[:, :, 0]

            # 2) READ in place.
            if sl is not None:
                return K[sl, :, :n], V[sl, :, :n]               # views: nothing is copied
            return K[lane_ids, :, :n], V[lane_ids, :, :n]       # one gather, n slots only

        # ---- prefill chunk: read first, then write ----------------------------------------
        # 1) READ the old ring and append the chunk. torch.cat allocates a NEW tensor, so the
        #    write below cannot corrupt what attention is about to read.
        old_k, old_v = (K[sl, :, :n], V[sl, :, :n]) if sl is not None else (K[lane_ids, :, :n], V[lane_ids, :, :n])
        k_all = torch.cat([old_k, k], dim=2)
        v_all = torch.cat([old_v, v], dim=2)

        # 2) WRITE only the last min(T, W) tokens (indices are then unique, so no write races)
        m = self._m
        K[lane_ids[:, None], :, phys] = k[:, :, T - m:].transpose(1, 2)
        V[lane_ids[:, None], :, phys] = v[:, :, T - m:].transpose(1, 2)

        return k_all, v_all