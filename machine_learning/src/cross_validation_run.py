from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch_geometric
from loguru import logger
from tqdm.auto import tqdm

from src.data_processing.data_loading import SamplePath
from src.dataset import Dataset, Sampler, TrainingDataset
from src.evaluation import Evaluator
from src.training import Trainer


@dataclass
class CrossValidationRun:
    initial_model: torch.nn.Module
    num_folds: int
    batch_size: int
    sample_path: SamplePath
    random_state: int
    num_workers: int

    synthesize_negative_samples: bool
    max_matrix_noise: float
    max_positive_upsampling_per_pathway: int

    max_matrix_nmae: float | None = None

    def __post_init__(self) -> None:
        logger.info("Generating Trainers")
        self.samples = self.sample_path.load_preprocessed_samples(
            max_matrix_nmae=self.max_matrix_nmae,
        )
        logger.info("Finished loading data")
        self._trainers = list(tqdm(self._generate_cv_trainers(), total=self.num_folds))
        logger.info("Finished initializing trainers")

    def run(
        self,
        num_epochs: int,
        with_tqdm: bool = True,  # noqa: FBT001, FBT002
        device: str = "cpu",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        training_metrics = []
        validation_results = []

        for fold, trainer in enumerate(self._trainers):
            logger.info(f"Training fold {fold+1}/{self.num_folds}")
            trainer.to(device)
            trainer_training_metrics, trainer_validation_results = (
                trainer.run_training_loop(num_epochs=num_epochs, with_tqdm=with_tqdm)
            )
            trainer.to("cpu")
            training_metrics.append(trainer_training_metrics.assign(fold=fold))
            validation_results.append(trainer_validation_results.assign(fold=fold))

        return pd.concat(training_metrics), pd.concat(validation_results)

    def _generate_cv_trainers(self) -> Iterator[Trainer]:
        for (
            training_dataloader,
            validation_dataloader,
        ) in self._generate_cv_dataloaders():
            model = copy.deepcopy(self.initial_model)
            yield Trainer(
                model=model,
                loss_function=torch.nn.BCEWithLogitsLoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=3e-3),
                training_dataloader=training_dataloader,
                evaluator=Evaluator(
                    model=model,
                    validation_dataloader=validation_dataloader,
                ),
            )

    def _generate_cv_dataloaders(
        self,
    ) -> Iterator[
        tuple[torch_geometric.loader.DataLoader, torch_geometric.loader.DataLoader]
    ]:
        for training_dataset, validation_dataset in self._generate_cv_datasets():
            yield self._get_training_dataloader(
                training_dataset,
            ), self._get_validation_dataloader(validation_dataset)

    def _get_training_dataloader(
        self,
        training_dataset: TrainingDataset,
    ) -> torch_geometric.loader.DataLoader:
        return torch_geometric.loader.DataLoader(
            dataset=training_dataset,
            batch_size=self.batch_size,
            exclude_keys=self.dataloader_keys_to_exclude,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.random_state),
            sampler=Sampler(
                dataset=training_dataset,
                shuffle=True,
                random_state=self.random_state,
                synthesize_negative_samples=self.synthesize_negative_samples,
            ),
        )

    def _get_validation_dataloader(
        self,
        validation_dataset: Dataset,
    ) -> torch_geometric.loader.DataLoader:
        return torch_geometric.loader.DataLoader(
            dataset=validation_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            exclude_keys=self.dataloader_keys_to_exclude,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.random_state),
            sampler=Sampler(
                dataset=validation_dataset,
                shuffle=False,
                random_state=self.random_state,
                synthesize_negative_samples=False,
            ),
        )

    def get_inference_dataloader(
        self,
        inference_sample_paths: SamplePath,
    ) -> torch_geometric.loader.DataLoader:
        return self._get_validation_dataloader(
            validation_dataset=Dataset(
                samples=inference_sample_paths.load_preprocessed_samples(),
            ),
        )

    @property
    def dataloader_keys_to_exclude(self) -> list[str]:
        return [
            "num_nodes",
            "compound_mats_unsmoothed",
            "node_is_primary",
            "new_experiment_data",
            "pathway_id",
            "node_names",
        ]

    def _generate_cv_datasets(self) -> Iterator[tuple[TrainingDataset, Dataset]]:
        for training_drugs, validation_drugs in self._generate_cv_drug_splits():
            yield (
                self._get_training_dataset(training_drugs=training_drugs),
                self._get_validation_dataset(validation_drugs=validation_drugs),
            )

    def _generate_cv_drug_splits(self) -> Iterator[tuple[set[str], set[str]]]:
        drug_names = np.array(list({sample.drug_name for sample in self.samples}))

        kfold = sklearn.model_selection.KFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        for train_indices, validation_indices in kfold.split(drug_names):
            yield set(drug_names[train_indices]), set(drug_names[validation_indices])

    def _get_training_dataset(self, training_drugs: set[str]) -> TrainingDataset:
        return TrainingDataset(
            [sample for sample in self.samples if sample.drug_name in training_drugs],
            random_state=self.random_state,
            synthesize_negative_samples=self.synthesize_negative_samples,
            max_matrix_noise=self.max_matrix_noise,
            max_positive_upsampling_per_pathway=self.max_positive_upsampling_per_pathway,
        )

    def _get_validation_dataset(self, validation_drugs: set[str]) -> Dataset:
        return Dataset(
            [sample for sample in self.samples if sample.drug_name in validation_drugs],
        )
