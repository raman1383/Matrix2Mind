import os
import json
import re

# Get the directory where tokenizer.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Build the path to the json file in the same directory
config_path = os.path.join(current_dir, "2048-tokenizer.json")

# 1. Load tokenizer configuration
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Reconstruct vocabulary and merge maps
vocab = {int(k): v for k, v in config["vocab"].items()}
char_to_id = {v: k for k, v in vocab.items() if len(v) == 1} # Only base chars



# Reconstruct merges map: dict of (int, int) -> int
merges = {}
for k, v in config["merges"].items():
    p1, p2 = map(int, k.split(","))
    merges[(p1, p2)] = v



# Special token info
special_tokens = config["special_tokens"]
special_token_id = list(special_tokens.values())[0]
special_token_str = list(special_tokens.keys())[0]


# decode must be able to print the special token even if it is not in `vocab`
id_to_str = dict(vocab)
id_to_str.setdefault(special_token_id, special_token_str)



class ID_mapper():

    @staticmethod
    def encode(text: str) -> list:
        """Split on the special token, BPE-encode the rest. Raises on unknown characters."""
        parts = re.split(rf"({re.escape(special_token_str)})", text)
        final_ids = []
        for part in parts:
            if part == special_token_str:
                final_ids.append(special_token_id)
            elif part:
                final_ids.extend(encode_chunk(part))
        return final_ids


    @staticmethod
    def decode(ids) -> str:
        """Token IDs -> string. Raises on IDs that are not in the vocabulary."""
        out = []
        for idx in ids:
            if idx not in id_to_str:
                raise ValueError(f"Token id {idx} is not in the vocabulary (size {len(id_to_str)})")
            out.append(id_to_str[idx])
        return "".join(out)



def merge_tokens(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def encode_chunk(text: str) -> list:
    """Encode text that contains no special tokens. Raises on characters outside the base vocab."""
    ids = []
    for pos, c in enumerate(text):
        if c not in char_to_id:
            raise ValueError(
                f"Character {c!r} (U+{ord(c):04X}) at position {pos} is not in the "
                f"tokenizer's base vocabulary"
            )
        ids.append(char_to_id[c])
 
    while len(ids) >= 2:
        pairs = list(zip(ids, ids[1:]))
        best_pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if best_pair not in merges:
            break
        ids = merge_tokens(ids, best_pair, merges[best_pair])
    return ids
