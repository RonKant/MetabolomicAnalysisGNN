from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_geometric


@dataclass
class SyntheticSampleGenerator:
    random_state: int
    matrix_bank: pd.DataFrame

    def __post_init__(self) -> None:
        self.numpy_rng = np.random.default_rng(seed=self.random_state)

    @classmethod
    def from_samples(
        cls,
        samples: list[torch_geometric.data.Data],
        **kwargs: dict[str, Any],
    ) -> "SyntheticSampleGenerator":
        return cls(
            matrix_bank=cls._extract_matrix_bank(samples=samples),
            **kwargs,
        )

    @classmethod
    def _extract_matrix_bank(
        cls,
        samples: list[torch_geometric.data.Data],
        use_original_matrices: bool = False,  # noqa: FBT001, FBT002
        filter_non_empty: bool = True,  # noqa: FBT001, FBT002
    ) -> pd.DataFrame:
        return (
            pd.concat(
                cls._extract_sample_matrix_bank(
                    sample,
                    use_original_matrices=use_original_matrices,
                )
                .query("~is_empty" if filter_non_empty else "(~is_empty) | is_empty")
                .assign(drug=sample.drug_name)
                for sample in samples
            )
            .drop(columns=["is_empty"])
            .drop_duplicates(subset=["drug", "compound_name"])
        )

    @classmethod
    def _extract_sample_matrix_bank(
        cls,
        sample: torch_geometric.data.Data,
        use_original_matrices: bool = False,  # noqa: FBT001, FBT002
    ) -> pd.DataFrame:
        sample_matrices = (
            sample.original_mats if use_original_matrices else sample.compound_mats
        )
        matrix_bank = pd.DataFrame(
            (
                (matrix, node_name.split("___")[0])
                for matrix, node_name in zip(sample_matrices, sample.node_names)
            ),
            columns=["matrix", "compound_name"],
        ).drop_duplicates(subset=["compound_name"])

        return matrix_bank.assign(
            is_empty=matrix_bank.matrix.apply(cls._is_matrix_empty),
        )

    @classmethod
    def _is_matrix_empty(cls, matrix: torch.Tensor) -> bool:
        return torch.allclose(matrix, torch.zeros_like(matrix))

    @torch.no_grad()
    def generate_negative_sample(
        self,
        base_sample: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        base_sample_compound_matrices = self._extract_sample_matrix_bank(base_sample)
        new_sample_compound_matrices = self._resample_matrices(
            base_sample_compound_matrices,
        )

        sample = self._get_sample_with_new_matrices(
            base_sample=base_sample,
            new_sample_compound_matrices=new_sample_compound_matrices,
        )
        sample.y = 0
        return sample

    def _resample_matrices(self, sample_matrix_bank: pd.DataFrame) -> pd.DataFrame:
        assert sample_matrix_bank.compound_name.nunique() == len(
            sample_matrix_bank.compound_name,
        )

        compounds_to_resample = sample_matrix_bank.query(
            "~is_empty",
        ).compound_name

        new_matrices = self.numpy_rng.choice(
            self.matrix_bank.matrix,
            size=len(compounds_to_resample),
            replace=False,
        )

        compound_to_new_matrix = dict(zip(compounds_to_resample, new_matrices))

        resampled_matrix_bank = sample_matrix_bank.copy()
        for (
            row_index,
            compound_name,
        ) in resampled_matrix_bank.compound_name.items():
            if compound_name in compound_to_new_matrix:
                resampled_matrix_bank.matrix.at[row_index] = compound_to_new_matrix[
                    compound_name
                ]

        return resampled_matrix_bank

    @classmethod
    def _get_sample_with_new_matrices(
        cls,
        base_sample: torch_geometric.data.Data,
        new_sample_compound_matrices: pd.DataFrame,
    ) -> torch_geometric.data.Data:
        new_sample_compound_matrices = new_sample_compound_matrices.set_index(
            "compound_name",
        )

        sample = base_sample.clone()

        sample.compound_mats = torch.stack(
            [
                new_sample_compound_matrices.matrix.loc[node_name.split("___")[0]]
                for node_name in sample.node_names
            ],
        )

        return sample
