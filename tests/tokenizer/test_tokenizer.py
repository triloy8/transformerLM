import pytest

from transformerlm.tokenizer.tokenizer import Tokenizer


def make_minimal_tokenizer():
    # Minimal byte-level vocab covering space, 'a', 'b', 'ab', and a special token
    vocab = {
        0: b" ",
        1: b"a",
        2: b"b",
        3: b"ab",  # merged token
        4: b"<|eot|>",
    }
    merges = [
        (b"a", b"b"),
    ]
    special_tokens = ["<|eot|>"]
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def test_roundtrip_basic_ascii_and_spaces():
    tok = make_minimal_tokenizer()
    text = "a b ab"
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert out == text


def test_special_tokens_are_atomic_and_roundtrip():
    tok = make_minimal_tokenizer()
    text = "a<|eot|>b"
    ids = tok.encode(text)
    # Expect exactly three tokens: 'a', special, 'b'
    assert len(ids) == 3
    dec = tok.decode(ids)
    assert dec == text


def test_merge_application_on_small_example():
    tok = make_minimal_tokenizer()
    ids = tok.encode("ab")
    # Should have a single token id corresponding to b"ab"
    assert len(ids) == 1


def test_encode_iterable_yields_consistent_stream():
    tok = make_minimal_tokenizer()
    lines = ["a b", "ab"]
    list_ids = list(tok.encode_iterable(lines))
    # Equivalent to concatenated encoding
    expected = tok.encode("".join(lines))
    assert list_ids == expected