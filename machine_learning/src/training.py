from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
import torch
import torch_geometric.loader
from tqdm.auto import tqdm

from src.batch_result import BatchResult
from src.evaluation import Evaluator


@dataclass
class Trainer:

    model: torch.nn.Module

    optimizer: torch.optim.Optimizer

    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    training_dataloader: torch_geometric.loader.DataLoader

    evaluator: Optional[Evaluator] = None  # noqa: FA100

    def run_training_loop(
        self,
        num_epochs: int,
        with_tqdm: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        epochs = range(num_epochs)
        if with_tqdm:
            epochs = tqdm(epochs)

        epoch_training_metrics = []
        epoch_validation_results = []

        for epoch in epochs:
            epoch_training_metrics.append(
                {**self._run_training_epoch(), "epoch": epoch},
            )

            if self.evaluator is not None:
                epoch_validation_results.append(
                    self.evaluator.run_validation_epoch().assign(epoch=epoch),
                )

        return (
            pd.DataFrame.from_records(epoch_training_metrics),
            pd.concat(epoch_validation_results),
        )

    def _run_training_epoch(self) -> pd.Series:
        self.model.train()

        device = next(iter(self.model.parameters())).device

        return BatchResult.combine(
            [
                self._do_gradient_step(batch.to(device))
                for batch in self.training_dataloader
            ],
        ).calculate_metrics()

    def _do_gradient_step(
        self,
        batch: torch_geometric.data.Batch,
    ) -> BatchResult:

        self.optimizer.zero_grad()

        logits = self.model(batch)

        loss = self.loss_function(input=logits.squeeze(), target=batch.y.float())

        loss.backward()

        self.optimizer.step()

        return BatchResult.from_batch(batch=batch, logits=logits)

    def to(self, device: torch.device) -> "Trainer":
        self.model.to(device)
        return self
