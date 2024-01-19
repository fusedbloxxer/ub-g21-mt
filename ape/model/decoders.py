import typing as t
import torch
import torch.nn as nn
from torch.nn import Embedding

from onmt.onmt.decoders.transformer import TransformerDecoderLayer, TransformerDecoderBase
from onmt.onmt.decoders.decoder import DecoderBase
from onmt.onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.onmt.modules.position_ffn import ActivationFunction
from onmt.onmt.utils.misc import sequence_mask
from onmt.onmt.modules.rmsnorm import RMSNorm
from onmt.onmt.modules.moe import MoE

from ..modules.multi_source_attn import SerialMHCrossAttentinon


class MultiSourceTransformerDecoderLayer(TransformerDecoderLayer):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    See https://tunz.kr/post/4 and :cite:`DeeperTransformer`.
    """
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        max_relative_positions=0,
        relative_positions_buckets=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        num_kv=0,
        add_ffnbias=True,
        parallel_residual=False,
        shared_layer_norm=False,
        layer_norm="standard",
        norm_eps=1e-6,
        use_ckpting=[],
        parallel_gpu=1,
        sliding_window=0,
        rotary_interleave=True,
        rotary_theta=1e4,
        rotary_dim=0,
        num_experts=0,
        num_experts_per_tok=2,
    ):
        """
        Args:
            See TransformerDecoderLayerBase
        """
        super(MultiSourceTransformerDecoderLayer, self).__init__(
            d_model,
            heads,
            d_ff,
            dropout,
            attention_dropout,
            self_attn_type,
            max_relative_positions,
            relative_positions_buckets,
            aan_useffn,
            full_context_alignment,
            alignment_heads,
            pos_ffn_activation_fn=pos_ffn_activation_fn,
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            add_ffnbias=add_ffnbias,
            parallel_residual=parallel_residual,
            shared_layer_norm=shared_layer_norm,
            layer_norm=layer_norm,
            norm_eps=norm_eps,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
            sliding_window=sliding_window,
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            rotary_dim=rotary_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

        self.context_attn = SerialMHCrossAttentinon(
            head_count=heads,
            model_dim=d_model,
            dropout=attention_dropout,
            self_attn_type=self.self_attn_type,
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )

        if layer_norm == "standard":
            self.layer_norm_2 = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm_2 = RMSNorm(d_model, eps=norm_eps)
        elif layer_norm == "identity":
            self.layer_norm_2 = nn.Identity()
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")

    def forward(self, *args, **kwargs):
        """Extend `_forward` for (possibly) multiple decoder pass:
        Always a default (future masked) decoder forward pass,
        Possibly a second future aware decoder pass for joint learn
        full context alignement, :cite:`garg2019jointly`.

        Args:
            * All arguments of _forward, of which
            with_align (bool): needed to compute attn_align
            return_attn (bool): to force MHA to return attns

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * layer_out ``(batch_size, T, model_dim)``
            * top_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None
        """
        # forward pass
        with_align = kwargs.pop("with_align", False)
        layer_out, (src_attns, mt_attns) = self._forward(*args, **kwargs)

        # compute top_attn for each source
        src_top_attn = None if src_attns is None else src_attns[:, 0, :, :].contiguous()
        mt_top_attn = None if src_attns is None else mt_attns[:, 0, :, :].contiguous()

        # optionally align attentions
        src_attn_align = None
        mt_attn_align = None

        # perform alignment
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, (src_attns, mt_attns) = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads > 0:
                src_attns = src_attns[:, : self.alignment_heads, :, :].contiguous()
                mt_attns = mt_attns[:, : self.alignment_heads, :, :].contiguous()

            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            src_attn_align = src_attns.mean(dim=1)
            mt_attn_align = mt_attns.mean(dim=1)
        return layer_out, (src_top_attn, mt_top_attn), (src_attn_align, mt_attn_align)

    def _forward(
        self,
        layer_in,
        src_enc_out,
        mt_enc_out,
        src_pad_mask,
        mt_pad_mask,
        tgt_pad_mask,
        step=None,
        future=False,
        return_attn=False,
    ) -> t.Tuple[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            layer_in (FloatTensor): ``(batch_size, T, model_dim)``
            src_enc_out (FloatTensor): ``(batch_size, src_len, model_dim)``
            mt_enc_out (FloatTensor): ``(batch_size, mt_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            mt_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.
            return_attn (bool) : if set True requires attns output

        Returns:
            (FloatTensor, FloatTensor):

            * layer_out ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None
        src_pad_mask = src_pad_mask.unsqueeze(1)  # [B, 1, 1, slen]
        mt_pad_mask = mt_pad_mask.unsqueeze(1)  # [B, 1, 1, slen]

        if layer_in.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)
            dec_mask = dec_mask.unsqueeze(1)
            dec_mask = dec_mask.expand(-1, -1, dec_mask.size(3), -1)
            src_pad_mask = src_pad_mask.expand(-1, -1, dec_mask.size(3), -1)
            mt_pad_mask = mt_pad_mask.expand(-1, -1, dec_mask.size(3), -1)
            # mask now are (batch x 1 x tlen x s or t len)
            # 1 = heads to be expanded in MHA

        norm_layer_in = self.layer_norm_1(layer_in)

        self_attn, _ = self._forward_self_attn(
            norm_layer_in, dec_mask, step, return_attn=return_attn
        )

        if self.dropout_p > 0:
            self_attn = self.dropout(self_attn)

        if self.parallel_residual:
            # perform cross attention
            ctx_attn, (src_attns, mt_attns) = self.context_attn.forward(
                query=norm_layer_in,
                src_key=src_enc_out,
                mt_key=mt_enc_out,
                src_value=src_enc_out,
                mt_value=mt_enc_out,
                src_mask=src_pad_mask,
                mt_mask=mt_pad_mask,
                return_attn=return_attn,
            )
            # feed_forward applies residual, so we remove and apply residual with un-normed
            layer_out = (
                self.feed_forward(norm_layer_in)
                - norm_layer_in
                + layer_in
                + self_attn
                + ctx_attn
            )
        else:
            query = self_attn + layer_in
            norm_query = self.layer_norm_2(query)
            ctx_attn, (src_attns, mt_attns) = self.context_attn.forward(
                query=norm_query,
                src_key=src_enc_out,
                mt_key=mt_enc_out,
                src_value=src_enc_out,
                mt_value=mt_enc_out,
                src_mask=src_pad_mask,
                mt_mask=mt_pad_mask,
                return_attn=return_attn,
            )
            if self.dropout_p > 0:
                ctx_attn = self.dropout(ctx_attn)
            layer_out = self.feed_forward(ctx_attn + query)

        return layer_out, (src_attns, mt_attns)


class MultiSourceTransformerBaseDecoder(TransformerDecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        num_layers (int): number of decoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, scaled-dot-flash, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        relative_positions_buckets (int):
            Number of buckets when using relative position bias
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear
        num_kv (int): number of heads for KV when different vs Q (multiquery)
        add_ffnbias (bool): whether to add bias to the FF nn.Linear
        parallel_residual (bool): Use parallel residual connections in each layer block, as used
            by the GPT-J and GPT-NeoX models
        shared_layer_norm (bool): When using parallel residual, share the input and post
            attention layer norms.
        layer_norm (string): type of layer normalization standard/rms
        norm_eps (float): layer norm epsilon
        use_ckpting (List): layers for which we checkpoint for backward
        parallel_gpu (int): Number of gpu for tensor parallelism
        sliding_window (int): Width of the band mask and KV cache (cf Mistral Model)
        rotary_interleave (bool): Interleave the head dimensions when rotary embeddings are applied
        rotary_theta (int): rotary base theta
        rotary_dim (int): in some cases the rotary dim is lower than head dim
        num_experts (int): Number of experts for MoE
        num_experts_per_tok (int): Number of experts choice per token
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        dropout,
        attention_dropout,
        embeddings: nn.Module,
        max_relative_positions,
        relative_positions_buckets,
        aan_useffn,
        full_context_alignment,
        alignment_layer,
        alignment_heads,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        num_kv=0,
        add_ffnbias=True,
        parallel_residual=False,
        shared_layer_norm=False,
        layer_norm="standard",
        norm_eps=1e-6,
        use_ckpting=[],
        parallel_gpu=1,
        sliding_window=0,
        rotary_interleave=True,
        rotary_theta=1e4,
        rotary_dim=0,
        num_experts=0,
        num_experts_per_tok=2,
    ):
        super(MultiSourceTransformerBaseDecoder, self).__init__(
            d_model,
            copy_attn,
            embeddings,
            alignment_layer,
            layer_norm,
            norm_eps,
        )

        # Use PyTorch Modules
        self.embeddings = t.cast(nn.Module, self.embeddings)

        # Decoder Transformer Layers with Multi-Headed Serial Cross-Attention
        self.transformer_layers = nn.ModuleList(
            [
                MultiSourceTransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                    relative_positions_buckets=relative_positions_buckets,
                    aan_useffn=aan_useffn,
                    full_context_alignment=full_context_alignment,
                    alignment_heads=alignment_heads,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    add_qkvbias=add_qkvbias,
                    num_kv=num_kv,
                    add_ffnbias=add_ffnbias,
                    parallel_residual=parallel_residual,
                    shared_layer_norm=shared_layer_norm,
                    layer_norm=layer_norm,
                    norm_eps=norm_eps,
                    use_ckpting=use_ckpting,
                    parallel_gpu=parallel_gpu,
                    sliding_window=sliding_window,
                    rotary_interleave=rotary_interleave,
                    rotary_theta=rotary_theta,
                    rotary_dim=rotary_dim,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                )
                for _ in range(num_layers)
            ]
        )

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(
        self,
        *,
        tgt_token_ids: torch.Tensor,
        src_enc_out: torch.Tensor,
        mt_enc_out: torch.Tensor,
        src_pad_mask: torch.Tensor,
        mt_pad_mask: torch.Tensor,
        step=None,
        **kwargs,
    ):
        """
        Decode, possibly stepwise.
        When training step is always None, when decoding, step increases.

        Args:
            tgt (Tensor): batch x tlen x feats
            src_enc_out (Tensor): src encoder output (batch x src_slen x model_dim)
            mt_enc_out (Tensor): mt encoder output (batch x mt_slen x model_dim)
        """
        # Perform alignment for SRC and MT cross-attentions maps
        with_align = kwargs.pop("with_align", False)
        return_attn = with_align or self._copy or kwargs.pop("return_attn", False)

        # Enable KV caching for decoding steps
        if step == 0:
            self._init_cache(src_enc_out)
        elif step is None:
            self._stop_cache()

        # Retrieve embeddings for decoder token_ids
        dec_out = self.embeddings(tgt_token_ids)
        pad_idx = self.embeddings.word_embeddings.padding_idx

        # Create a tensor to mask the padding
        tgt_pad_mask = tgt_token_ids.eq(pad_idx).unsqueeze(1)  # [B x 1 x T_tgt]
        src_pad_mask = src_pad_mask.unsqueeze(1) # [B x 1 x slen]
        mt_pad_mask = mt_pad_mask.unsqueeze(1) # [B x 1 x slen]

        # Track attention alignments
        src_attn_aligns = []
        mt_attn_aligns = []
        src_top_attn = None
        mt_top_attn = None

        # Forward pass through all decoder layers
        for layer in self.transformer_layers:
            dec_out, (src_top_attn, mt_top_attn), (src_attn_align, mt_attn_align) = layer(
                layer_in=dec_out,
                src_enc_out=src_enc_out,
                mt_enc_out=mt_enc_out,
                src_pad_mask=src_pad_mask,
                mt_pad_mask=mt_pad_mask,
                tgt_pad_mask=tgt_pad_mask,
                with_align=with_align,
                return_attn=return_attn,
                step=step,
            )

            if src_attn_align is not None and mt_attn_align is not None:
                src_attn_aligns.append(src_attn_align)
                mt_attn_aligns.append(mt_attn_align)
        dec_out = self.layer_norm(dec_out)

        # Aggregate attention information
        src_attns = { "std": src_top_attn }
        mt_attns = { "std": mt_top_attn }
        if self._copy:
            src_attns["copy"] = src_top_attn
            mt_attns["copy"] = mt_top_attn
        if with_align:
            src_attns["align"] = src_attn_aligns[self.alignment_layer] # `(B, Q, K)`
            mt_attns["align"] = mt_attn_aligns[self.alignment_layer] # `(B, Q, K)`
        return dec_out, (src_attns, mt_attns)

    def _init_cache(self, enc_out):
        batch_size = enc_out.size(0)
        depth = enc_out.size(-1)

        for layer in self.transformer_layers:
            # first value set to True triggered by the beginning of decoding
            # layer_cache becomes active in the MultiHeadedAttention fwd
            layer.context_attn.cross_attn_src.layer_cache = (
                True,
                {
                    "keys": torch.tensor([], device=enc_out.device),
                    "values": torch.tensor([], device=enc_out.device),
                },
            )
            layer.context_attn.cross_attn_mt.layer_cache = (
                True,
                {
                    "keys": torch.tensor([], device=enc_out.device),
                    "values": torch.tensor([], device=enc_out.device),
                },
            )

            if isinstance(layer.self_attn, AverageAttention):
                layer.self_attn.layer_cache = True, {
                    "prev_g": torch.zeros(
                        (batch_size, 1, depth), device=enc_out.device
                    ).to(enc_out.dtype)
                }
            else:
                layer.self_attn.layer_cache = (
                    True,
                    {
                        "keys": torch.tensor([], device=enc_out.device),
                        "values": torch.tensor([], device=enc_out.device),
                    },
                )
                if hasattr(layer.self_attn, "rope"):
                    layer.self_attn.rope = layer.self_attn.rope.to(enc_out.device)
                    layer.self_attn.cos = layer.self_attn.cos.to(enc_out.device)
                    layer.self_attn.sin = layer.self_attn.sin.to(enc_out.device)

    def _stop_cache(self):
        for layer in self.transformer_layers:
            if isinstance(layer.self_attn, AverageAttention):
                layer.self_attn.layer_cache = False, {"prev_g": torch.tensor([])}
            else:
                layer.self_attn.layer_cache = (
                    False,
                    {"keys": torch.tensor([]), "values": torch.tensor([])},
                )
            layer.context_attn.layer_cache = (
                False,
                {"keys": torch.tensor([]), "values": torch.tensor([])},
            )


class MultiSourceTransformerDecoder(MultiSourceTransformerBaseDecoder):
    def __init__(
        self,
        embeddings: nn.Module,
        num_layers=3,
        d_model=768,
        heads=2,
        d_ff=2048,
        copy_attn=False,
        self_attn_type='scaled-dot-flash',
        dropout=0.1,
        attention_dropout=0.1,
        max_relative_positions=0,
        relative_positions_buckets=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_layer=None,
        alignment_heads=None,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        num_kv=0,
        add_ffnbias=True,
        parallel_residual=False,
        shared_layer_norm=False,
        layer_norm="standard",
        norm_eps=1e-5,
        use_ckpting=[],
        parallel_gpu=1,
        sliding_window=0,
        rotary_interleave=True,
        rotary_theta=10000,
        rotary_dim=0,
        num_experts=0,
        num_experts_per_tok=2,
    ) -> None:
        """Create a MultiSourceTransformerDecoder.

        Args:
            embeddings (_type_): _description_
        """
        super(MultiSourceTransformerDecoder, self).__init__(
            num_layers,
            d_model,
            heads,
            d_ff,
            copy_attn,
            self_attn_type,
            dropout,
            attention_dropout,
            embeddings,
            max_relative_positions,
            relative_positions_buckets,
            aan_useffn,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            pos_ffn_activation_fn,
            add_qkvbias,
            num_kv,
            add_ffnbias,
            parallel_residual,
            shared_layer_norm,
            layer_norm, norm_eps,
            use_ckpting,
            parallel_gpu,
            sliding_window,
            rotary_interleave,
            rotary_theta,
            rotary_dim,
            num_experts,
            num_experts_per_tok
        )

        # Perform weight initialization
        self.init()

    def init(self) -> None:
        for name, module in self.named_modules():
            if name.startswith('embeddings'):
                continue
            for name, param in module.named_parameters():
                if   name == 'weight' and param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
                elif name == 'bias':
                    torch.nn.init.zeros_(param)
                else:
                    continue

    def forward(
        self,
        tgt_input_ids: torch.Tensor,
        src_enc_out: torch.Tensor,
        mt_enc_out: torch.Tensor,
        src_pad_mask: torch.Tensor,
        mt_pad_mask: torch.Tensor,
        step: t.Optional[int]=None,
        return_attn: bool=False,
        with_align: bool=False,
        **kwargs,
    ):
        # Forward pass through the decoder
        dec_out, (src_attns, mt_attns) = super().forward(
            tgt_token_ids=tgt_input_ids,
            src_enc_out=src_enc_out,
            mt_enc_out=mt_enc_out,
            src_pad_mask=src_pad_mask,
            mt_pad_mask=mt_pad_mask,
            return_attn=return_attn,
            with_align=with_align,
            step=step,
            **kwargs,
        )

        return dec_out, (src_attns, mt_attns)
