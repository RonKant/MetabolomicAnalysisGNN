from dataclasses import dataclass

import pandas as pd
import torch
import torch_geometric

from src.batch_result import BatchResult


@dataclass
class Evaluator:

    model: torch.nn.Module

    validation_dataloader: torch_geometric.loader.DataLoader

    @torch.no_grad()
    def run_validation_epoch(self) -> pd.DataFrame:
        self.model.eval()

        all_batch_results = [
            BatchResult.from_batch(batch=batch, logits=self.model(batch))
            for batch in self.validation_dataloader
        ]
        return BatchResult.combine(all_batch_results).to_dataframe()
