import typing as t
from typing import List, Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
from abc import ABC, abstractmethod
from functools import partial

from onmt.onmt.modules.multi_headed_attn import MultiHeadedAttention


class MultiSourceMHCrossAttention(ABC, nn.Module):
    def __init__(
        self,
        head_count: int,
        model_dim: int,
        dropout: float = 0.1,
        self_attn_type: t.Optional[str] = None,
        add_qkvbias: bool = False,
        num_kv: int = 0,
        use_ckpting: t.List[str] = [],
        parallel_gpu: int = 1,
    ) -> None:
        super(MultiSourceMHCrossAttention, self).__init__()

        # Both MHAttentions follow the same structure
        self.__create_mha = partial(
            MultiHeadedAttention,
            head_count=head_count,
            model_dim=model_dim,
            dropout=dropout,
            self_attn_type=self_attn_type,
            add_qkvbias=add_qkvbias,
            attn_type="context",
            num_kv=num_kv,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )

        # Need one Multi Headed Cross Attention for each source input
        self.cross_attn_src = self.__create_mha()
        self.cross_attn_mt = self.__create_mha()

    @abstractmethod
    def forward(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        mask: Optional[Tensor] = None,
        sliding_window: Optional[int] = 0,
        step: Optional[int] = 0,
        return_attn: Optional[bool] = False,
        self_attn_type: Optional[str] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError()

    def update_dropout(self, dropout: float) -> None:
        self.cross_attn_src.update_dropout(dropout)
        self.cross_attn_mt.update_dropout(dropout)


class SerialMHCrossAttentinon(MultiSourceMHCrossAttention):
    """
    Serial Strategy:
        "Input Combination Strategies for Multi-Source Transformer Decoder"
        (Jindrich et al.)
    """
    def __init__(
        self,
        head_count: int,
        model_dim: int,
        dropout: float = 0.1,
        self_attn_type: str | None = None,
        add_qkvbias: bool = False,
        num_kv: int = 0,
        use_ckpting: List[str] = [],
        parallel_gpu: int = 1,
    ) -> None:
        super(SerialMHCrossAttentinon, self).__init__(
            head_count,
            model_dim,
            dropout,
            self_attn_type,
            add_qkvbias,
            num_kv,
            use_ckpting,
            parallel_gpu,
        )

    def forward(
        self,
        query: Tensor,
        src_key: Tensor,
        mt_key: Tensor,
        src_value: Tensor,
        mt_value: Tensor,
        src_mask: Tensor | None = None,
        mt_mask: Tensor | None = None,
        sliding_window: int | None = 0,
        step: int | None = 0,
        return_attn: bool | None = False,
        self_attn_type: str | None = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Compute the context vector and the attention vectors.

        Args:
            query (Tensor): set of `query_len`
                query vectors  ``(batch, query_len, dim)``
            src_key (Tensor): set of `src_key_len`
                key vectors ``(batch, src_key_len, dim)``
            mt_key (Tensor): set of `mt_key_len`
                key vectors ``(batch, mt_key_len, dim)``
            src_value (Tensor): set of `src_key_len`
                value vectors ``(batch, src_key_len, dim)``
            mt_value (Tensor): set of `mt_key_len`
                value vectors ``(batch, mt_key_len, dim)``
            src_mask: binary mask 1/0 indicating which keys have
                zero / non-zero attention ``(batch, query_len, src_key_len)``
            mt_mask: binary mask 1/0 indicating which keys have
                zero / non-zero attention ``(batch, query_len, mt_key_len)``
            step (int): decoding step (used for Rotary embedding)
        Returns:
            (Tensor, Tensor):

            * output context vectors ``(batch, query_len, dim)``
            * SRC and MT Attention vectors in heads
                (``(batch, head, query_len, src_key_len)``, ``(batch, head, query_len, mt_key_len)``)
        """
        # Attend to the SRC encoding
        src_ctx_attn, src_attns = self.cross_attn_src.forward(
            key=src_key,
            value=src_value,
            query=query,
            mask=src_mask,
            sliding_window=sliding_window,
            step=step,
            return_attn=return_attn,
            self_attn_type=self_attn_type,
        )

        # Add residual connection
        query = src_ctx_attn + query

        # Attend to the MT encoding
        mt_ctx_attn, mt_attns = self.cross_attn_mt.forward(
            key=mt_key,
            value=mt_value,
            query=query,
            mask=mt_mask,
            sliding_window=sliding_window,
            step=step,
            return_attn=return_attn,
            self_attn_type=self_attn_type,
        )

        # Add residual connection
        query = mt_ctx_attn + query

        # Return the output and the two cross-attentions
        return query, (src_attns, mt_attns)
