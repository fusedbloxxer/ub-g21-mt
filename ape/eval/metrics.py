from typing import TypedDict, Dict, Any, cast
import evaluate as eval
from evaluate import Metric
from evaluate import EvaluationModule
from evaluate import EvaluationModuleInfo
from dataclasses import dataclass, field
from datasets import Features, Sequence, Value
from pathlib import Path


class BLEUMetricOutput(TypedDict):
    bleu: Any
    precisions: Any
    brevity_penalty: Any
    length_ratio: Any
    translation_length: Any
    reference_length: Any


class CHRFMetricOutput(TypedDict):
    score: Any
    char_order: Any
    word_order: Any
    beta: Any


class TERMetricOutput(TypedDict):
    score: Any
    num_edits: Any
    ref_length: Any


@dataclass
class APEMetricsOutput(object):
    bleu: BLEUMetricOutput
    chrf: CHRFMetricOutput
    ter: TERMetricOutput


class APEMetrics(Metric):
    def __init__(self, cache_dir: Path, **kwargs):
        super(APEMetrics, self).__init__(num_process=1, process_id=0, **kwargs)
        self._load_metrics(str(cache_dir / 'metrics'))

    def compute(self, *, predictions=None, references=None, **kwargs) -> APEMetricsOutput:
        """Compute the evaluation module.

        Usage of positional arguments is not allowed to prevent mistakes.

        Args:
            predictions (`list/array/tensor`, *optional*):
                Predictions.
            references (`list/array/tensor`, *optional*):
                References.
            **kwargs (optional):
                Keyword arguments that will be forwarded to the evaluation module [`~evaluate.EvaluationModule.compute`]
                method (see details in the docstring).

        Return:
            `dict` or `None`

            - Dictionary with the results if this evaluation module is run on the main process (`process_id == 0`).
            - `None` if the evaluation module is not run on the main process (`process_id != 0`).

        ```py
        >>> import evaluate
        >>> accuracy =  evaluate.load("accuracy")
        >>> accuracy.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 1])
        ```
        """
        output = super().compute(predictions=predictions, references=references, **kwargs)
        return cast(APEMetricsOutput, output)

    def _compute(self, predictions, references) -> APEMetricsOutput:
        # Wrap references in lists
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        # Compute each metric
        bleu_output = self.bleu.compute(predictions=predictions, references=references)
        chrf_output = self.chrf.compute(predictions=predictions, references=references)
        ter_output = self.ter.compute(predictions=predictions, references=references)

        # Aggregate all metrics
        return APEMetricsOutput(
            bleu=cast(BLEUMetricOutput, bleu_output),
            chrf=cast(CHRFMetricOutput, chrf_output),
            ter=cast(TERMetricOutput, ter_output),
        )

    def _load_metrics(self, cache_dir: str) -> None:
        # Bilingual Evaluation Understudy
        # https://huggingface.co/spaces/evaluate-metric/bleu
        self.bleu = eval.load('bleu', cache_dir=cache_dir)

        # CHaRacter-level F-score
        # https://huggingface.co/spaces/evaluate-metric/chrf
        self.chrf = eval.load('chrf', cache_dir=cache_dir)

        # Translation Error Rate (effort score => lower is better)
        # https://huggingface.co/spaces/evaluate-metric/ter
        self.ter = eval.load('ter', cache_dir=cache_dir)

    def _info(self) -> EvaluationModuleInfo:
        return EvaluationModuleInfo(
            reference_urls=['https://www2.statmt.org/wmt23/ape-task.html'],
            description='APE Metrics containing BLEU, TER and CHRF',
            citation='Shared Task: Automatic Post-Editing (WMT23)',
            features=Features({
                'predictions': Value('string'),
                'references': Sequence(Value('string')),
            })
        )
