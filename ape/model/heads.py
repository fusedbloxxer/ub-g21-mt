import typing as t
import torch
import torch.nn as nn
from torch.nn.utils import skip_init

from onmt.onmt.modules.position_ffn import ACTIVATION_FUNCTIONS
from onmt.onmt.modules.position_ffn import ActivationFunction
from onmt.onmt.modules.rmsnorm import RMSNorm


class MultiSourceLMHead(nn.Module):
    def __init__(
        self,
        add_bias: bool,
        layer_norm: str,
        hidden_size: int,
        word_embeddings: nn.Embedding,
        activation_fun: str=ActivationFunction.relu,
        norm_eps: float=1e-5,
    ) -> None:
        super(MultiSourceLMHead, self).__init__()

        # add one hidden layer
        self.dense = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=hidden_size,
            bias=add_bias,
        )
        self.activ_fn = ACTIVATION_FUNCTIONS[activation_fun]

        # then perform normalization
        if layer_norm == "standard":
            self.layer_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(hidden_size, eps=norm_eps)
        elif layer_norm == "identity":
            self.layer_norm = nn.Identity()
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")

        # and project back the input to vocab using the same word_embeddings
        self.decoder = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=word_embeddings.weight.shape[0],
            bias=True,
        )

        # Initialize the weights
        self.init()

        # Tie decoder to vocab embeddings
        self.vocab_size: int = word_embeddings.weight.size(0)
        self.tie_weights(word_embeddings)

    def init(self) -> None:
        for name, module in self.named_modules():
            if name.startswith('decoder'):
                torch.nn.init.zeros_(module.bias)
            for name, param in module.named_parameters():
                if   name == 'weight' and param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
                elif name == 'bias':
                    torch.nn.init.zeros_(param)
                else:
                    continue

    def tie_weights(self, vocab_embeddings: nn.Embedding) -> None:
        self.decoder.weight = nn.Parameter(vocab_embeddings.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.dense(features)
        x = self.activ_fn(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x
