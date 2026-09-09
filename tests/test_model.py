import torch

from tiny_chat.config import ModelConfig
from tiny_chat.model import ChatModel, parameter_count
from tiny_chat.tokenizer import ByteBPETokenizer
from tiny_chat.chat import clean_response


def test_bpe_round_trip_and_compression():
    tokenizer = ByteBPETokenizer.train("hello hello hello", vocab_size=260)
    text = "hello hello"
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text
    assert len(tokens) < len(text.encode("utf-8"))


def test_clean_response_removes_role_boundaries():
    assert clean_response("Assistant: hello\nAssistant: leaked\nUser: next") == "hello"


def test_model_shape_and_size():
    config = ModelConfig()
    model = ChatModel(config)
    logits, loss, cache = model(torch.zeros((2, 16), dtype=torch.long), torch.zeros((2, 16), dtype=torch.long))
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None
    assert len(cache) == config.layers
    assert cache[0][0].shape[1] == config.key_value_heads
    assert 18_000_000 <= parameter_count(model) <= 22_000_000


def test_cached_generation_keeps_growing_sequence():
    model = ChatModel(ModelConfig())
    output = model.generate(torch.zeros((1, 4), dtype=torch.long), max_new_tokens=3, top_k=4, top_p=0.9)
    assert output.shape == (1, 7)
