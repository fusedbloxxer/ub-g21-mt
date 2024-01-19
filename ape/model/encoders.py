import typing as t
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel


class MultiSourceTransformerEncoder(nn.Module):
    def __init__(self, *, encoder_src: PreTrainedModel, encoder_mt: PreTrainedModel) -> None:
        super(MultiSourceTransformerEncoder, self).__init__()
        self.encoder_src = encoder_src
        self.encoder_mt = encoder_mt

    def forward(self,
                src_input_ids: Tensor,
                mt_input_ids: Tensor,
                src_attn_mask: Tensor,
                mt_attn_mask: Tensor):
        # Encode inputs from different languages using different encoders
        src_encoding = self.encoder_src.forward(input_ids=src_input_ids,
                                                attention_mask=src_attn_mask,
                                                return_dict=True)
        mt_encoding = self.encoder_mt.forward(input_ids=mt_input_ids,
                                              attention_mask=mt_attn_mask,
                                              return_dict=True)

        # Extract the hidden activation for each token present in the sequences
        return src_encoding.last_hidden_state, mt_encoding.last_hidden_state
