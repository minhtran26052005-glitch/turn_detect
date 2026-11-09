from turn_detection.codec import encode_pair, normalize, validate_future


class Tokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def encode(self, value, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(10, 10 + len(value.split())))


def test_normalize_preserves_words_but_removes_case_and_punctuation():
    assert normalize("  XIN, Chào! Ａ  ") == "xin chào a"


def test_encoding_preserves_future_and_newest_context():
    encoded = encode_pair(Tokenizer(), "one two three four five", "next word", max_length=8)
    assert encoded == {"input_ids": [1, 12, 13, 14, 2, 10, 11, 2], "attention_mask": [1] * 8}


def test_future_word_count():
    assert validate_future("next") == ["next"]
    assert validate_future("next word") == ["next", "word"]
    import pytest
    with pytest.raises(ValueError):
        validate_future("one two three")
