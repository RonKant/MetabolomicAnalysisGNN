import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch_geometric

from src.data_processing.data_loading import get_metadata
from src.data_processing.synthetic_data_generation import SyntheticSampleGenerator


@dataclass
class SamplingIndex:
    sample_index: int
    synthesize_negative: bool


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[torch_geometric.data.Data]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: SamplingIndex) -> torch_geometric.data.Data:
        assert not idx.synthesize_negative
        return self._transform(self._samples[idx.sample_index])

    def _transform(
        self,
        sample: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        sample = sample.clone()
        sample.compound_mats = (
            torch.log10(torch.clip(sample.compound_mats, min=10**3)) - 3
        ) / 6  # for normalizing in range [0,1] (approx)
        return sample

    @property
    def metadata(self) -> pd.DataFrame:
        return get_metadata(self._samples)


class TrainingDataset(Dataset):
    def __init__(
        self,
        samples: list[torch_geometric.data.Data],
        random_state: int,
        max_matrix_noise: float,
        synthesize_negative_samples: bool,  # noqa: FBT001
        max_positive_upsampling_per_pathway: float,
    ) -> None:
        self.max_positive_upsampling_per_pathway = max_positive_upsampling_per_pathway
        samples_upsampled = self._oversample_positives(
            self._oversample_positives_by_pathway(samples),
        )
        super().__init__(samples=samples_upsampled)
        self.random_state = random_state
        self.numpy_rng = np.random.default_rng(seed=self.random_state)
        self.max_matrix_noise = max_matrix_noise
        if synthesize_negative_samples:
            self._synthetic_sample_generator = SyntheticSampleGenerator.from_samples(
                samples,
                random_state=random_state,
            )
        else:
            self._synthetic_sample_generator = None

    @classmethod
    def _oversample_positives(
        cls,
        samples: list[torch_geometric.data.Data],
    ) -> list[torch_geometric.data.Data]:
        negative_samples = [sample for sample in samples if sample.y == 0]
        positive_samples = [sample for sample in samples if sample.y != 0]
        sample_ratio = int(
            np.ceil(
                (len(negative_samples) + len(positive_samples)) / len(positive_samples),
            ),
        )

        return [*negative_samples, *(positive_samples * sample_ratio)]

    def _oversample_positives_by_pathway(
        self,
        samples: list[torch_geometric.data.Data],
    ) -> list[torch_geometric.data.Data]:
        oversampling_ratios = self._get_sample_oversampling_ratio(samples)

        return list(
            itertools.chain(
                *(
                    [samples[row.sample_index]] * row.upsampling_ratio
                    for _, row in oversampling_ratios.iterrows()
                ),
            ),
        )

    def _get_sample_oversampling_ratio(
        self,
        samples: list[torch_geometric.data.Data],
    ) -> pd.DataFrame:
        metadata = get_metadata(samples)
        pathway_oversampling_ratio = self._get_pathway_oversampling_ratio(samples)
        sample_oversampling_ratio = (
            metadata.reset_index()
            .rename(columns={"index": "sample_index"})
            .merge(
                pathway_oversampling_ratio.assign(label=1),
                on=["pathway", "label"],
                how="outer",
            )
            .fillna(1)
        )
        sample_oversampling_ratio.loc[
            sample_oversampling_ratio.label == 0,
            "upsampling_ratio",
        ] = 1
        return sample_oversampling_ratio.loc[
            :,
            ["sample_index", "upsampling_ratio"],
        ].astype(int)

    def _get_pathway_oversampling_ratio(
        self,
        samples: list[torch_geometric.data.Data],
    ) -> pd.DataFrame:
        metadata = get_metadata(samples)
        positive_label_statistics = (
            metadata.groupby("pathway")
            .agg(
                positive_samples=("label", "sum"),
                total_samples=("label", "size"),
            )
            .query("positive_samples > 0")
        )
        positive_label_statistics = positive_label_statistics.assign(
            positive_ratio=(
                positive_label_statistics.positive_samples
                / positive_label_statistics.total_samples
            ),
        )
        return positive_label_statistics.assign(
            upsampling_ratio=np.ceil(
                positive_label_statistics.positive_ratio.max()
                / positive_label_statistics.positive_ratio,
            ).clip(lower=1, upper=self.max_positive_upsampling_per_pathway),
        ).loc[:, ["upsampling_ratio"]]

    def __getitem__(self, idx: SamplingIndex) -> torch_geometric.data.Data:
        if idx.synthesize_negative:
            assert self._synthetic_sample_generator is not None
            base_sample = self._samples[idx.sample_index]
            synthetic_sample = (
                self._synthetic_sample_generator.generate_negative_sample(
                    base_sample=base_sample,
                )
            )
            return self._transform(synthetic_sample)
        return super().__getitem__(idx)

    def _transform(
        self,
        sample: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        sample = self._random_flip(sample)
        sample = self._random_noise(sample)
        return super()._transform(sample)

    def _random_flip(
        self,
        sample: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        if self.numpy_rng.random() < 0.5:  # noqa: PLR2004
            sample = sample.clone()
            sample.compound_mats = sample.compound_mats.flip(-1, -2)

        return sample

    @torch.no_grad()
    def _random_noise(
        self,
        sample: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        if self.max_matrix_noise is None or self.max_matrix_noise == 0:
            return sample

        sample = sample.clone()
        noise_level = self.numpy_rng.uniform(
            low=1 - self.max_matrix_noise,
            high=1 + self.max_matrix_noise,
            size=sample.compound_mats.shape,
        )
        sample.compound_mats *= torch.Tensor(noise_level)
        return sample


class Sampler:
    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool,  # noqa: FBT001
        random_state: int,
        synthesize_negative_samples: bool,  # noqa: FBT001
    ) -> None:
        self.metadata = dataset.metadata
        self.shuffle = shuffle
        self.numpy_rng = np.random.default_rng(seed=random_state)
        self.synthesize_negative_samples = synthesize_negative_samples

    def __len__(self) -> int:
        return len(self.metadata)

    def __iter__(self) -> Iterator[SamplingIndex]:
        sample_indices = self._get_sample_indices()

        if self.shuffle:
            self.numpy_rng.shuffle(sample_indices)

        yield from sample_indices

    def _get_sample_indices(self) -> pd.DataFrame:
        sample_indices_dataframe = pd.concat(
            [self._get_positive_sample_indices(), self._get_negative_sample_indices()],
        )
        return [
            SamplingIndex(**row.to_dict())
            for _, row in sample_indices_dataframe.iterrows()
        ]

    def _get_positive_sample_indices(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_index": self.metadata.query("label == 1").index,
                "synthesize_negative": False,
            },
        )

    def _get_negative_sample_indices(self) -> pd.DataFrame:
        if self.synthesize_negative_samples:
            return pd.DataFrame(
                {
                    "sample_index": self.numpy_rng.choice(
                        self.metadata.index,
                        (self.metadata.label == 0).sum(),
                    ),
                    "synthesize_negative": True,
                },
            )
        return pd.DataFrame(
            {
                "sample_index": self.metadata.query("label == 0").index,
                "synthesize_negative": False,
            },
        )
