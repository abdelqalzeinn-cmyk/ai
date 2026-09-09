import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.key_value_heads = config.key_value_heads
        self.head_repeats = config.head_repeats
        self.head_size = config.head_size
        self.query_key_value = nn.Linear(
            config.embedding_size,
            (config.heads + 2 * config.key_value_heads) * self.head_size,
        )
        self.output = nn.Linear(config.embedding_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)
        inverse_frequency = 1.0 / (10000 ** (torch.arange(0, self.head_size, 2).float() / self.head_size))
        self.register_buffer("inverse_frequency", inverse_frequency)

    def _apply_rope(self, values: torch.Tensor, start: int) -> torch.Tensor:
        positions = torch.arange(start, start + values.size(2), device=values.device)
        angles = torch.outer(positions.float(), self.inverse_frequency).to(values.device)
        cosine = angles.cos()[None, None, :, :]
        sine = angles.sin()[None, None, :, :]
        first, second = values[..., ::2], values[..., 1::2]
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
        return values * cosine.repeat_interleave(2, dim=-1) + rotated * sine.repeat_interleave(2, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, length, channels = x.shape
        query_size = self.heads * self.head_size
        key_value_size = self.key_value_heads * self.head_size
        query, key, value = self.query_key_value(x).split((query_size, key_value_size, key_value_size), dim=-1)
        query = query.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        key = key.view(batch, length, self.key_value_heads, self.head_size).transpose(1, 2)
        value = value.view(batch, length, self.key_value_heads, self.head_size).transpose(1, 2)
        past_length = 0 if past_key_value is None else past_key_value[0].size(2)
        query = self._apply_rope(query, past_length)
        key = self._apply_rope(key, past_length)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        present = (key, value)
        key = key.repeat_interleave(self.head_repeats, dim=1)
        value = value.repeat_interleave(self.head_repeats, dim=1)
        attention = (query @ key.transpose(-2, -1)) * (self.head_size ** -0.5)
        if past_key_value is None:
            mask = torch.triu(torch.ones(length, length, device=x.device), diagonal=1).bool()
            attention = attention.masked_fill(mask, float("-inf"))
        attention = self.dropout(F.softmax(attention, dim=-1))
        output = attention @ value
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.dropout(self.output(output)), present


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.embedding_size)
        self.attention = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.embedding_size)
        self.gate = nn.Linear(config.embedding_size, config.intermediate_size)
        self.up = nn.Linear(config.embedding_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attention, present = self.attention(self.norm1(x), past_key_value)
        x = x + attention
        feed_forward = self.down(F.silu(self.gate(self.norm2(x))) * self.up(self.norm2(x)))
        return x + self.dropout(feed_forward), present


class ChatModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = RMSNorm(config.embedding_size)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor | None = None, past_key_values=None):
        hidden = self.token_embedding(inputs)
        presents = []
        for block, past in zip(self.blocks, past_key_values or [None] * len(self.blocks)):
            hidden, present = block(hidden, past)
            presents.append(present)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, presents

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        past_key_values = None
        for _ in range(max_new_tokens):
            context = inputs if past_key_values is None else inputs[:, -1:]
            logits, _, past_key_values = self(context, past_key_values=past_key_values)
            next_logits = logits[:, -1] / max(temperature, 1e-3)
            if repetition_penalty != 1.0:
                for token in inputs.unique().tolist():
                    next_logits[:, token] = torch.where(
                        next_logits[:, token] < 0,
                        next_logits[:, token] * repetition_penalty,
                        next_logits[:, token] / repetition_penalty,
                    )
            if top_k > 0:
                top_k = min(top_k, next_logits.size(-1))
                threshold = torch.topk(next_logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            if 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                sorted_probabilities = F.softmax(sorted_logits, dim=-1)
                cumulative = sorted_probabilities.cumsum(dim=-1)
                remove = cumulative - sorted_probabilities > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                next_logits = torch.full_like(next_logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
