from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.cross_validation_run import CrossValidationRun
from src.model import PathwayDrugResponseEncoder


@dataclass
class Experiment:
    runs: dict[str, CrossValidationRun]
    results_directory: Path

    def run_and_save_results(
        self,
        num_epochs: int,
        with_tqdm: bool,  # noqa: FBT001
        device: str = "cpu",
    ) -> None:
        logger.info(f"Run names: {list(self.runs.keys())}")
        for run_name, run in self.runs.items():
            logger.info(f"Running {run_name}")
            training_metrics_path = self.get_training_metrics_path(run_name=run_name)
            val_results_path = self.get_validation_results_path(run_name=run_name)
            logger.info(f"Results will be saved to: {training_metrics_path}")
            logger.info(f"Results will be saved to: {val_results_path}")
            training_metrics, validation_results = run.run(
                num_epochs=num_epochs,
                with_tqdm=with_tqdm,
                device=device,
            )
            logger.info("Saving results...")
            training_metrics.to_csv(
                training_metrics_path,
                index=False,
            )
            validation_results.to_csv(
                val_results_path,
                index=False,
            )
            logger.info("Saving complete")

    def get_training_metrics_path(self, run_name: str) -> Path:
        return (self.results_directory / (run_name + "_train.csv")).resolve()

    def get_validation_results_path(self, run_name: str) -> Path:
        return (self.results_directory / (run_name + "_val.csv")).resolve()


@dataclass
class SingleParameterExperimentCreation:
    default_model_parameters: dict[str, Any]
    default_experiment_parameters: dict[str, Any]
    changing_parameter_name: str
    changing_parameter_values: list[Any]
    is_model_parameter: bool
    experiment_name_prefix: str
    results_directory: Path

    def create_experiment(
        self,
    ) -> Experiment:
        run_generator = (
            self._generate_model_parameter_runs()
            if self.is_model_parameter
            else self._generate_experiment_parameter_runs()
        )

        return Experiment(
            runs=dict(
                zip(
                    self._generate_run_names(),
                    run_generator,
                ),
            ),
            results_directory=self.results_directory,
        )

    def _generate_run_names(self) -> Iterator[str]:
        yield from (
            self.experiment_name_prefix + run_name_suffix
            for run_name_suffix in self._generate_run_name_suffixes()
        )

    def _generate_run_name_suffixes(self) -> Iterator[str]:
        yield from (
            f"_{self.changing_parameter_name}_{experiment_parameter_value}"
            for experiment_parameter_value in self.changing_parameter_values
        )

    def _generate_experiment_parameter_runs(self) -> Iterator[CrossValidationRun]:
        for experiment_parameter_value in self.changing_parameter_values:
            yield CrossValidationRun(
                initial_model=PathwayDrugResponseEncoder(
                    **self.default_model_parameters,
                ),
                **{
                    **self.default_experiment_parameters,
                    self.changing_parameter_name: experiment_parameter_value,
                },
            )

    def _generate_model_parameter_runs(self) -> Iterator[CrossValidationRun]:
        for experiment_parameter_value in self.changing_parameter_values:
            yield CrossValidationRun(
                initial_model=PathwayDrugResponseEncoder(
                    **{
                        **self.default_model_parameters,
                        self.changing_parameter_name: experiment_parameter_value,
                    },
                ),
                **self.default_experiment_parameters,
            )
