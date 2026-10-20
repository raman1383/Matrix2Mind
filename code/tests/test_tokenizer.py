# tests/test_tokenizer.py
#
# TEST 7: the BPE tokenizer must be lossless.
# decode(encode(s)) == s for anything made of base-vocabulary characters, the special
# token must survive as ONE id, and bad input must fail loudly instead of corrupting data.

import random

import pytest

import tokenizerModule as tm
from tokenizerModule import ID_mapper

BASE_CHARS = sorted(tm.char_to_id)     # every character the tokenizer can represent


def random_text(seed: int, n: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(BASE_CHARS) for _ in range(n))


@pytest.mark.parametrize("seed", range(20))
def test_round_trip_random_text(seed):
    s = random_text(seed, n=10 + 15 * seed)
    ids = ID_mapper.encode(s)
    assert ID_mapper.decode(ids) == s
    assert len(ids) <= len(s)                              # merges can only shorten it


def test_round_trip_realistic_text():
    sample = (
        "Once upon a time, there was a little girl named Lily.\n"
        "She loved to play with her dog in the park, and they were very happy.\n"
    )
    s = "".join(c for c in sample if c in tm.char_to_id)    # keep only representable chars
    assert len(s) > 50
    ids = ID_mapper.encode(s)
    assert ID_mapper.decode(ids) == s
    assert len(ids) < len(s)                               # real text must actually compress


def test_empty_string():
    assert ID_mapper.encode("") == []
    assert ID_mapper.decode([]) == ""


def test_ids_are_in_range():
    ids = ID_mapper.encode(random_text(0, 500))
    assert min(ids) >= 0
    assert max(ids) <= max(tm.id_to_str)


def test_first_merge_is_applied():
    # the earliest-learned merge combines two base characters, so those two characters
    # side by side must become exactly the one merged id
    (a, b), new_id = min(tm.merges.items(), key=lambda kv: kv[1])
    text = tm.vocab[a] + tm.vocab[b]
    assert ID_mapper.encode(text) == [new_id]


def test_special_token_is_one_id_and_round_trips():
    x, y = BASE_CHARS[0], BASE_CHARS[1]
    eos = tm.special_token_str

    assert ID_mapper.encode(eos) == [tm.special_token_id]

    s = f"{x}{y}{eos}{y}{x}{eos}{eos}"
    ids = ID_mapper.encode(s)
    assert ids.count(tm.special_token_id) == 3
    assert ID_mapper.decode(ids) == s


def test_unknown_character_raises():
    emoji = "\U0001F600"
    if emoji in tm.char_to_id:
        pytest.skip("this tokenizer happens to know the emoji")
    with pytest.raises(ValueError, match="not in the"):
        ID_mapper.encode(f"{BASE_CHARS[0]}{emoji}")


def test_decode_unknown_id_raises():
    with pytest.raises(ValueError, match="not in the vocabulary"):
        ID_mapper.decode([10**6])