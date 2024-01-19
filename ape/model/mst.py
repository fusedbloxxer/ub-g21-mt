import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from transformers import PreTrainedModel
from collections import defaultdict

from onmt.onmt.modules.position_ffn import ActivationFunction

from .encoders import MultiSourceTransformerEncoder
from .decoders import MultiSourceTransformerDecoder
from .heads import MultiSourceLMHead
from .types import CausalLMOutput


class MultiSourceTransformerCausalLM(nn.Module):
    def __init__(
        self,
        *,
        block_size: int=512,
        encoder_mt: PreTrainedModel,
        encoder_src: PreTrainedModel,
    ) -> None:
        super(MultiSourceTransformerCausalLM, self).__init__()
        self.block_size = block_size

        # SRC and MT Encoder
        self.encoder = MultiSourceTransformerEncoder(
            encoder_src=encoder_src,
            encoder_mt=encoder_mt,
        )

        # Multi Source Decoder
        self.decoder = MultiSourceTransformerDecoder(
            pos_ffn_activation_fn=ActivationFunction.silu,
            self_attn_type='scaled-dot-flash',
            embeddings=encoder_mt.embeddings,
            parallel_residual=False,
            layer_norm='standard',
            norm_eps=1e-5,
            num_layers=3,
            heads=4,
        )

        # Language Modeling Head
        self.lm_head = MultiSourceLMHead(
            word_embeddings=encoder_mt.embeddings.word_embeddings,
            activation_fun=ActivationFunction.relu,
            layer_norm='standard',
            hidden_size=768,
            add_bias=True,
            norm_eps=1e-5,
        )

    def forward(
        self,
        *,
        src_input_ids: torch.Tensor,
        mt_input_ids: torch.Tensor,
        src_attn_mask: torch.Tensor,
        mt_attn_mask: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        labels: torch.Tensor | None=None,
        step: int | None=None,
        **kwargs,
    ) -> CausalLMOutput:
        # Forward pass through SRC and MT encoders
        src_last_hidden_state, mt_last_hidden_state = self.encoder.forward(
            src_input_ids=src_input_ids,
            mt_input_ids=mt_input_ids,
            src_attn_mask=src_attn_mask,
            mt_attn_mask=mt_attn_mask,
        )

        # Forward pass through the causal decoder language model
        dec_out, _ = self.decoder.forward(
            tgt_input_ids=tgt_input_ids,
            src_enc_out=src_last_hidden_state,
            mt_enc_out=mt_last_hidden_state,
            src_pad_mask=src_attn_mask,
            mt_pad_mask=mt_attn_mask,
            step=step,
            **kwargs,
        )

        # Project last_hidden_state back to vocabulary dimension
        pred_scores = self.lm_head.forward(dec_out)

        # Optimize for causal language modeling objective
        if labels is not None:
            pred = pred_scores[:, :-1, :].contiguous().view(-1, self.lm_head.vocab_size)
            true = labels[:, 1:].contiguous().view(-1)
            lm_loss = cross_entropy(pred, true)
        else:
            lm_loss = None

        # Return token_ids and loss
        return CausalLMOutput(
            logits=pred_scores,
            loss=lm_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        src_input_ids: torch.Tensor,
        mt_input_ids: torch.Tensor,
        src_attn_mask: torch.Tensor,
        mt_attn_mask: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        max_new_tokens,
        temperature=1.0,
        do_sample=False,
        top_k=None,
    ):
        """ Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
            the sequence max_new_tokens times, feeding the predictions back into the model each time.
            Most likely you'll want to make sure to be in model.eval() mode of operation for this.
            (from: https://github.com/karpathy/minGPT)
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = tgt_input_ids if tgt_input_ids.size(1) <= self.block_size else tgt_input_ids[:, -self.block_size:]

            # forward the model to get the logits for the index in the sequence
            logits = self.forward(
                src_input_ids=src_input_ids,
                mt_input_ids=mt_input_ids,
                src_attn_mask=src_attn_mask,
                mt_attn_mask=mt_attn_mask,
                tgt_input_ids=idx_cond,
            ).logits

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            tgt_input_ids = torch.cat((tgt_input_ids, idx_next), dim=1)

        return tgt_input_ids

    @torch.no_grad()
    def post_edit(
        self,
        *,
        src_input_ids: torch.Tensor,
        mt_input_ids: torch.Tensor,
        src_attn_mask: torch.Tensor,
        mt_attn_mask: torch.Tensor,
        bos_token_id: int | None,
        temperature=1.0,
        do_sample=False,
        top_k=None,
    ):
        # Create a batch of sentences with only one BOS `token_id` [B x 1]
        tgt_seq_ids = torch.full((src_input_ids.size(0), 1), fill_value=bos_token_id if bos_token_id else 0, device=src_input_ids.device)

        # Generate using guiding from cross-attention encodings
        tgt_token_ids = self.generate(
            src_input_ids=src_input_ids,
            mt_input_ids=mt_input_ids,
            src_attn_mask=src_attn_mask,
            mt_attn_mask=mt_attn_mask,
            tgt_input_ids=tgt_seq_ids,
            max_new_tokens=src_input_ids.size(1),
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
        )

        # (B, T) output token_ids
        return tgt_token_ids
