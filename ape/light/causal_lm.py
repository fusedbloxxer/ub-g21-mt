import typing as t
import torch
import torch.nn
from torch.optim import Adam
import lightning as lit
from lightning.pytorch.loggers import MLFlowLogger
from transformers import AutoModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from dataclasses import asdict
from flatten_dict import flatten

from .. import HF_CACHE_DIR, DEVICE
from ..eval.metrics import APEMetrics
from ..model.mst import MultiSourceTransformerCausalLM


class MultiSourceCausalLMLightningModule(lit.LightningModule):
    def __init__(
        self,
        *,
        block_size: int,
        encoder_type_mt: str,
        encoder_type_src: str,
        tokenizer_mt: PreTrainedTokenizer | PreTrainedTokenizerFast,
        bos_token_id: int | None,
        temperature=1.0,
        do_sample=True,
        top_k=40,
    ) -> None:
        super(MultiSourceCausalLMLightningModule, self).__init__()
        self.logger: MLFlowLogger

        # Generation settings
        self.tokenizer_mt = tokenizer_mt
        self.bos_token_id = bos_token_id
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_k = top_k

        # Load source and target pretrained encoders
        encoder_src = AutoModel.from_pretrained(encoder_type_src, add_pooling_layer=False, cache_dir=HF_CACHE_DIR / 'models')
        encoder_mt = AutoModel.from_pretrained(encoder_type_mt, add_pooling_layer=False, cache_dir=HF_CACHE_DIR / 'models')

        # Combine encoders with a transformer encoder through cross-attention
        self.model = MultiSourceTransformerCausalLM(encoder_src=encoder_src, encoder_mt=encoder_mt, block_size=block_size)

        # Metrics used for validation
        self.valid_metrics = APEMetrics(cache_dir=HF_CACHE_DIR)
        self.test_metrics = APEMetrics(cache_dir=HF_CACHE_DIR)

    def training_step(self, batch, batch_idx):
        # Perform Causal Language Modeling
        output = self.model.forward(
            src_input_ids=batch['src_input_ids'],
            mt_input_ids=batch['mt_input_ids'],
            src_attn_mask=batch['src_attention_mask'],
            mt_attn_mask=batch['mt_attention_mask'],
            tgt_input_ids=batch['pe_input_ids'],
            labels=batch['pe_input_ids'],
        )

        # This is a training step => we have loss to compute
        assert output.loss is not None, 'could not compute loss'

        # Log progress
        self.logger.log_metrics(
            {
                'train_loss': output.loss.detach().cpu().item(),
            },
        step=self.global_step)

        # Forward loss for backprop
        return output.loss

    def validation_step(self, batch, batch_idx):
        # Perform Causal Language Modeling
        output = self.model.post_edit(
            src_input_ids=batch['src_input_ids'],
            mt_input_ids=batch['mt_input_ids'],
            src_attn_mask=batch['src_attention_mask'],
            mt_attn_mask=batch['mt_attention_mask'],
            bos_token_id=self.bos_token_id,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_k=self.top_k,
        )

        # Decode using MT tokenizer because they share the same vocabulary
        pred_sequences = self.tokenizer_mt.batch_decode(output, skip_special_tokens=True)
        true_sequences = self.tokenizer_mt.batch_decode(batch['pe_input_ids'], skip_special_tokens=True)
        true_sequences = [[sample] for sample in true_sequences]

        # Update metrics
        self.valid_metrics.add_batch(
            predictions=pred_sequences,
            references=true_sequences,
        )

    def on_validation_end(self) -> None:
        # Flatten dict and keep only valid items that may be logged
        metrics = flatten(asdict(self.valid_metrics.compute()), reducer='dot')
        metrics.pop('bleu.precisions')

        # Prefix with current phase
        metrics = { f'valid.{key}': value for key, value in metrics.items() }

        # Keep track of progress
        self.logger.log_metrics(metrics)

    def test_step(self, batch, batch_idx):
        # Perform Causal Language Modeling
        output = self.model.post_edit(
            src_input_ids=batch['src_input_ids'],
            mt_input_ids=batch['mt_input_ids'],
            src_attn_mask=batch['src_attention_mask'],
            mt_attn_mask=batch['mt_attention_mask'],
            bos_token_id=self.bos_token_id,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_k=self.top_k,
        )

        # Decode using MT tokenizer because they share the same vocabulary
        pred_sequences = self.tokenizer_mt.batch_decode(output, skip_special_tokens=True)
        true_sequences = self.tokenizer_mt.batch_decode(batch['pe_input_ids'], skip_special_tokens=True)
        true_sequences = [[sample] for sample in true_sequences]

        # Update metrics
        self.test_metrics.add_batch(
            predictions=pred_sequences,
            references=true_sequences,
        )

    def on_test_end(self) -> None:
        # Flatten dict and keep only valid items that may be logged
        metrics = flatten(asdict(self.test_metrics.compute()), reducer='dot')
        metrics.pop('bleu.precisions')

        # Prefix with current phase
        metrics = { f'test.{key}': value for key, value in metrics.items() }

        # Keep track of progress
        self.logger.log_metrics(metrics)

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.998))
        return opt
