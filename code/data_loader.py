import numpy as np
import torch
from pathlib import Path


def convert_txt_to_bin(txt_path: str | Path, bin_path: str | Path) -> None:
    """
    One-time conversion:  whitespace-separated token IDs -> uint16 binary file.
    Vocab size 2048 easily fits in uint16.
    """
    txt_path = Path(txt_path)
    bin_path = Path(bin_path)

    tokens = np.fromfile(txt_path, dtype=np.int64, sep=" ")

    if tokens.size == 0:
        raise ValueError(f"No token IDs found in {txt_path}")
    if tokens.min() < 0:
        raise ValueError(f"Negative token ID found in {txt_path}")
    if tokens.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"Token ID {tokens.max()} does not fit in uint16")

    tokens = tokens.astype(np.uint16)

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(bin_path)

    print(f"Converted {txt_path} -> {bin_path} ({tokens.size:,} tokens)")


class BatchLoader:
    """
    Reads a uint16 token file through a memmap.

    get_batch()          random windows (training); own RNG, deterministic given `seed`
    sequential_batches() fixed, non-overlapping windows (evaluation)
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int,
        seq_len: int,
        device,
        seed: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        self.tokens = self._load_tokens(self.data_path)

        if len(self.tokens) < self.seq_len + 1:
            raise ValueError(
                f"{self.data_path} contains only {len(self.tokens)} tokens, "
                f"but seq_len={self.seq_len} requires at least {self.seq_len + 1}."
            )

        # Private RNG: never touches the global PyTorch RNG.
        # so data order is reproducible and independent of the global RNG.
        self.rng = torch.Generator(device="cpu")
        if seed is None:
            self.rng.seed()
        else:
            self.rng.manual_seed(seed)

        # Reused for every batch. Shape: [1, seq_len + 1]
        self.offsets = torch.arange(self.seq_len + 1, dtype=torch.long).unsqueeze(0)

    @staticmethod
    def _load_tokens(path: Path) -> np.memmap:
        if not path.exists():
            raise FileNotFoundError(f"Token file not found: {path}")
        size = path.stat().st_size
        if size == 0:
            raise ValueError(f"Token file is empty: {path}")
        if size % np.dtype(np.uint16).itemsize != 0:
            raise ValueError(f"Invalid uint16 .bin file size: {path}")
        return np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.tokens)

    # ------------------------------------------------------------------
    def get_batch(self):
        """Random windows for training."""
        max_start = len(self.tokens) - self.seq_len - 1
        starts = torch.randint(
            low=0,
            high=max_start + 1,
            size=(self.batch_size,),
            generator=self.rng,
            device="cpu",
        )
        return self._make_batch(starts)

    def _make_batch(self, starts: torch.Tensor):
        """
        starts: [B] -> x, y of shape [B, seq_len] on `device`.
        One indexed read of seq_len+1 tokens gives both x and y.
        """
        starts = starts.to(dtype=torch.long)
        window_indices = starts.unsqueeze(1) + self.offsets          # [B, seq_len+1]
        windows = self.tokens[window_indices.numpy()].astype(np.int64)
        windows = torch.from_numpy(windows)

        x = windows[:, :-1].to(self.device)
        y = windows[:, 1:].to(self.device)
        return x, y

    def sequential_batches(self, max_batches: int | None = None):
        """
        Deterministic evaluation pass. Window starts are 0, seq_len, 2*seq_len, ...
        so every window scores a distinct set of target tokens, in the same order
        on every call. The last batch may be smaller than batch_size.
        """
        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be >= 0 or None")

        max_start = len(self.tokens) - self.seq_len - 1
        starts = torch.arange(0, max_start + 1, self.seq_len, dtype=torch.long)

        num_batches = 0
        for i in range(0, len(starts), self.batch_size):
            if max_batches is not None and num_batches >= max_batches:
                break
            yield self._make_batch(starts[i : i + self.batch_size])
            num_batches += 1

    # ------------------------------------------------------------------
    # for reproducible checkpointing
    def state_dict(self) -> dict:
        return {"rng_state": self.rng.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self.rng.set_state(state["rng_state"].cpu())