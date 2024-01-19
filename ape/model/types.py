from dataclasses import dataclass
from typing import Optional
from torch import Tensor


@dataclass
class CausalLMOutput(object):
    loss: Optional[Tensor]
    logits: Tensor
