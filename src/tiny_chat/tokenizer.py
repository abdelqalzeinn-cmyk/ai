from collections import Counter


class ByteBPETokenizer:
    """Small byte-level BPE tokenizer with lossless UTF-8 fallback."""

    base_vocab_size = 256

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = merges or []
        self.merge_to_id = {pair: self.base_vocab_size + index for index, pair in enumerate(self.merges)}
        self.token_bytes = {index: bytes([index]) for index in range(self.base_vocab_size)}
        for index, (left, right) in enumerate(self.merges):
            self.token_bytes[self.base_vocab_size + index] = self.token_bytes[left] + self.token_bytes[right]

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, max_chars: int = 1_000_000):
        if not 256 <= vocab_size <= 4096:
            raise ValueError("vocab_size must be between 256 and 4096")
        sequence = list(text[:max_chars].encode("utf-8", errors="ignore"))
        merges = []
        for _ in range(vocab_size - cls.base_vocab_size):
            pair_counts = Counter(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            (left, right), count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            merges.append((left, right))
            new_id = cls.base_vocab_size + len(merges) - 1
            sequence = cls._replace_pair(sequence, (left, right), new_id)
        return cls(merges)

    @staticmethod
    def _replace_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        result = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                result.append(new_id)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    def encode(self, text: str) -> list[int]:
        sequence = list(text.encode("utf-8", errors="replace"))
        for pair, token_id in self.merge_to_id.items():
            sequence = self._replace_pair(sequence, pair, token_id)
        return sequence

    def decode(self, tokens: list[int]) -> str:
        raw = b"".join(self.token_bytes.get(token, b"?") for token in tokens)
        return raw.decode("utf-8", errors="replace")

    def state_dict(self) -> dict:
        return {"merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_state_dict(cls, state: dict):
        return cls([tuple(pair) for pair in state["merges"]])
