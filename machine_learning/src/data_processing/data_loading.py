from __future__ import annotations

import pickle
import random
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch_geometric
from loguru import logger
from tqdm.auto import tqdm

SampleList = list[torch_geometric.data.Data]
SampleFilterFunction = Callable[[SampleList], SampleList]

MIN_NUM_NONZERO_COMPOUNDS = 5


class SamplePath(Enum):
    old_smoothing = "old_smoothing"
    new_smoothing = "new_smoothing"
    unsmoothed = "unsmoothed"
    unfiltered = "unfiltered"
    no_graph_data = "no_graph_data"
    empty = "empty"
    new_experiment = "new_experiment"

    def load_preprocessed_samples(
        self,
        max_matrix_nmae: float | None = None,
    ) -> SampleList:
        samples = self.load_raw_samples()

        for filter_function in self.filter_functions:
            samples = filter_function(samples, max_matrix_nmae=max_matrix_nmae)

        return samples

    @property
    def filter_functions(self) -> list[SampleFilterFunction]:
        match self:
            case SamplePath.new_experiment:
                return [_filter_max_matrix_nmae, _filter_empty_samples]
            case _:
                return [
                    _filter_used_drugs,
                    _filter_max_matrix_nmae,
                    _filter_empty_samples,
                    # has to be last:
                    _filter_drugs_with_positive_samples,
                ]

    def load_raw_samples(self) -> SampleList:
        return pickle.loads(
            self.path.read_bytes(),
        )

    @property
    def path(self) -> Path:
        logger.info(self)
        match self:
            case SamplePath.old_smoothing:
                return Path(
                    "/home/ronkanto/data/machine_learning/data/ds_orig_ae_flipped_correctly.dat",
                )
            case SamplePath.new_smoothing:
                return Path(
                    "/home/ronkanto/data/machine_learning/data/ds_orig_ae_flipped_correctly_resmoothed.dat",
                )
            case SamplePath.unsmoothed:
                return Path(
                    "/home/ronkanto/data/machine_learning/data/ds_orig_ae_flipped_correctly_unsmoothed_filled.dat",
                )
            case SamplePath.unfiltered:
                return Path(
                    "/home/ronkanto/data/machine_learning/data/ds_unfiltered.dat",
                )
            case SamplePath.no_graph_data:
                return Path(
                    "/home/ronkanto/data/machine_learning/data/ds_orig_ae_flipped_correctly_no_graph_data.dat",
                )
            case SamplePath.empty:
                return Path("/home/ronkanto/data/machine_learning/data/ds_empty.dat")
            case SamplePath.new_experiment:
                return Path(
                    "/home/ronkanto/data/machine_learning/data/ds_new_experiment.dat",
                )
            case _:
                msg = "Invalid sample path"
                raise RuntimeError(msg)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed.seed_everything(seed)


def get_metadata(samples: SampleList) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "drug": sample.drug_name,
                "pathway": sample.pathway_name,
                "label": sample.y,
            }
            for sample in samples
        ],
    )


def _filter_used_drugs(dataset: SampleList, **kwargs) -> SampleList:
    used_drugs = set(
        pd.read_csv(
            Path("/home/ronkanto/data/machine_learning/data/used_drugs.csv"),
        ).drug_name,
    )

    all_drugs_in_raw_data = {sample.drug_name for sample in dataset}

    assert all(drug in all_drugs_in_raw_data for drug in used_drugs)

    return [sample for sample in dataset if sample.drug_name in used_drugs]


def _filter_empty_samples(samples: SampleList, **kwargs) -> SampleList:
    def _get_num_nonzero_compounds(sample: torch_geometric.data.Data) -> int:
        is_nonzero = pd.DataFrame(
            zip(sample.node_names, sample.compound_mats.any(dim=1).any(dim=1).numpy()),
            columns=["node_name", "is_nonzero"],
        )
        is_nonzero = is_nonzero.assign(
            compound_name=is_nonzero.node_name.str.split("___").apply(
                lambda parts: parts[0],
            ),
        ).drop_duplicates(subset=["compound_name"])
        return is_nonzero.is_nonzero.sum()

    return [
        sample
        for sample in samples
        if _get_num_nonzero_compounds(sample) >= MIN_NUM_NONZERO_COMPOUNDS
    ]


def _filter_max_matrix_nmae(
    samples: SampleList,
    max_matrix_nmae: float | None = None,
) -> SampleList:
    if max_matrix_nmae is None:
        return samples

    def _nmae(smooth_matrix: torch.Tensor, original_matrix: torch.Tensor) -> float:
        return (
            (smooth_matrix - original_matrix).abs().sum() / original_matrix.abs().sum()
        ).item()

    filtered_samples = []

    for sample in tqdm(samples):
        for smooth_matrix, original_matrix in zip(
            sample.compound_mats,
            sample.original_mats,
        ):
            new_sample = sample.clone()
            new_sample.compound_mats = torch.stack(
                [
                    (
                        torch.zeros_like(smooth_matrix)
                        if (smooth_matrix == 0).all()
                        or (
                            _nmae(
                                smooth_matrix=smooth_matrix,
                                original_matrix=original_matrix,
                            )
                            > max_matrix_nmae
                        )
                        else smooth_matrix
                    )
                    for smooth_matrix, original_matrix in zip(
                        sample.compound_mats,
                        sample.original_mats,
                    )
                ],
            )

        filtered_samples.append(new_sample)

    logger.info(f"filtered {len(filtered_samples)} by nmae")

    return filtered_samples


def _filter_drugs_with_positive_samples(samples: SampleList, **kwargs) -> SampleList:
    metadata = get_metadata(samples)
    drugs_with_positive_samples = set(
        metadata.query("label == 1").drug.value_counts().index,
    )

    drugs_without_positive_samples = set(metadata.drug) - drugs_with_positive_samples
    logger.info(f"drugs without positive samples: {drugs_without_positive_samples}")
    return [
        sample for sample in samples if sample.drug_name in drugs_with_positive_samples
    ]
