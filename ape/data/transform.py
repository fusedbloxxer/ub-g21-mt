import torch
from torch import Tensor
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from typing import Any, List, Dict, Set, cast
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, BatchEncoding

from .types import APETripletDict


@dataclass
class Tokenizer(ABC):
    source_prefix: str = field(kw_only=True, default='')

    def __call__(self, x: List[str]) -> Dict[str, Tensor]:
        # Tokenize batch of text into `token_ids`
        batch_encoding = self.tokenize(x)

        # Append `_` if `source_prefix` is given
        if not self.source_prefix:
            prefix = ''
        else:
            prefix = '{}_'.format(self.source_prefix)

        # Prefix each output by `source_prefix` to overcome conflicts
        return { f'{prefix}{key}': value for key, value in batch_encoding.items() }

    def tokenize(self, x: List[str]) -> Dict[str, Tensor]:
        raise NotImplementedError()


@dataclass
class HFTokenizer(Tokenizer):
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    max_length: int = field(default=512, kw_only=True)

    def tokenize(self, text: List[str]) -> Dict[str, Tensor]:
        # Tokenize the input
        batch_encoding = self.tokenizer(text=text,
                                                       truncation=True,
                                                       padding='longest',
                                                       return_tensors='pt',
                                                       add_special_tokens=True,
                                                       max_length=self.max_length,
                                                       return_attention_mask=True)

        # Mark zeros as `True`, they will be masked later
        batch_encoding['attention_mask'] = torch.eq(cast(Tensor, batch_encoding['attention_mask']), 0)

        return cast(Dict[str, Tensor], batch_encoding)


@dataclass(init=False)
class Tokenize(object):
    def __init__(self, tokenizers: List[Tokenizer]) -> None:
        # Keep one tokenizer per `source_prefix`
        self.tokenizers: Dict[str, Tokenizer] = dict()

        # Enforce uniqueness between tokenizers
        for tknzr in tokenizers:
            if tknzr.source_prefix in self.tokenizers:
                raise ValueError(f'found duplicate source_prefix in different tokenizers: {tknzr.source_prefix}')
            else:
                self.tokenizers[tknzr.source_prefix] = tknzr

    def __call__(self, data: APETripletDict[List[str]] | List[APETripletDict[str]]) -> Dict[str, Tensor]:
        if isinstance(data, list):
            # List[Dict] -> Dict[List]
            data_copy = defaultdict(list)
            for entry in data:
                for source, text in entry.items():
                    data_copy[source].append(text)

            # Disallow defaults from now
            data_copy.default_factory = None
            data = cast(APETripletDict[List[str]], data_copy)

        # Aggregate outputs from multiple sources
        context = {}
        for source, entries in data.items():
            # Ensure that the source can be handled
            if source not in self.tokenizers:
                raise ValueError(f'missing tokenizer for source of type: {source}')

            # Tokenize and add output to context
            output = self.tokenizers[source](cast(List[str], entries))
            context.update(output)
        return context
