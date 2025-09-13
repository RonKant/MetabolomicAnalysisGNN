import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import sklearn.metrics
import torch
import torch_geometric


@dataclass
class BatchResult:
    drug: list[str]
    pathway: list[str]
    label: list[bool]
    logit: list[float]

    @classmethod
    def from_batch(
        cls,
        batch: torch_geometric.data.Batch,
        logits: torch.Tensor,
    ) -> "BatchResult":
        return BatchResult(
            drug=batch.drug_name,
            pathway=batch.pathway_name,
            label=batch.y.tolist(),
            logit=logits.squeeze().tolist(),
        )

    @staticmethod
    def combine(
        batch_results: Sequence["BatchResult"],
    ) -> "BatchResult":
        return BatchResult(
            **{
                key: list(
                    itertools.chain(
                        *(vars(batch_result)[key] for batch_result in batch_results),
                    ),
                )
                for key in vars(batch_results[0])
            },
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(vars(self)).sort_values(by=["drug", "pathway"])

    def calculate_metrics(self) -> dict[str, float]:
        return {
            metric_name: metric_function()
            for metric_name, metric_function in self.metric_functions.items()
        }

    @property
    def metric_functions(self) -> dict[str, Callable[None, float]]:
        return {"bce": self.get_bce, "accuracy": self.get_accuracy, "auc": self.get_auc}

    def get_bce(self) -> float:
        return torch.nn.BCEWithLogitsLoss()(
            input=torch.Tensor(self.logit),
            target=torch.Tensor(self.label),
        ).item()

    def get_accuracy(self) -> float:
        return sklearn.metrics.accuracy_score(
            y_true=self.label,
            y_pred=[logit > 0 for logit in self.logit],
        )

    def get_auc(self) -> float:
        return sklearn.metrics.roc_auc_score(
            y_true=self.label,
            y_score=self.logit,
        )
