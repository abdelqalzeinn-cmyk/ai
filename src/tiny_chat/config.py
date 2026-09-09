from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 512
    context_size: int = 256
    embedding_size: int = 512
    layers: int = 6
    heads: int = 8
    key_value_heads: int = 2
    intermediate_size: int = 1792
    dropout: float = 0.1

    @property
    def head_size(self) -> int:
        return self.embedding_size // self.heads

    @property
    def head_repeats(self) -> int:
        return self.heads // self.key_value_heads
