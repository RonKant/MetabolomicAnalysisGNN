from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import functools

import torch
import torch_geometric

from src.data_processing.data_loading import set_seed


class PoolingLayerEnum(Enum):
    mean_pool = 0
    sum_pool = 1

    def get_layer(self) -> torch.nn.Module:
        match self:
            case PoolingLayerEnum.mean_pool:
                return torch_geometric.nn.aggr.MeanAggregation()
            case PoolingLayerEnum.sum_pool:
                return torch_geometric.nn.aggr.SumAggregation()
            case _:
                msg = "Invalid layer"
                raise RuntimeError(msg)


class PathwayDrugResponseEncoder(torch.nn.Module):
    def __init__(
        self,
        convolution_channels: list,
        num_graph_layers: int,
        random_state: int,
        use_batchnorm: bool,  # noqa: FBT001
        dropout_probability: float | None,
        gnn_pooling_layer: PoolingLayerEnum,
        gnn_activation: str,
        encoder_type: MatrixEncoderType,
        activation_type: MatrixEncoderActivationType,
        gnn_network_type: str,
    ) -> None:
        super().__init__()
        set_seed(seed=random_state)
        self.dropout_probability = dropout_probability
        self._drug_matrix_encoder = ResponseMatrixEncoder(
            convolution_channels=convolution_channels,
            use_batchnorm=use_batchnorm,
            dropout_probability=dropout_probability,
            encoder_type=encoder_type,
            activation=activation_type.activation_fn,
        )

        self._pathway_encoder = PathwayGNNEncoder(
            num_graph_layers=num_graph_layers,
            pooling_layer=gnn_pooling_layer,
            activation=gnn_activation,
            network_type=gnn_network_type,
        )

    def forward(self, batch: torch_geometric.data.Batch) -> torch.Tensor:
        batch = batch.to(self.device)

        matrix_encodings = self._drug_matrix_encoder(batch.compound_mats)
        return self._pathway_encoder(batch=batch, matrix_encodings=matrix_encodings)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


@dataclass
class MatrixEncoderType:
    data_type: str = "full"
    row_number: int | None = None

    def __post_init__(self) -> None:
        allowed_rows = range(6)
        match self.data_type:
            case "full" | "diagonal":
                assert self.row_number is None
            case "row" | "column":
                assert self.row_number in allowed_rows
            case _:
                msg = "Invalid MatrixEncoderType {self}"
                raise RuntimeError(msg)

    def __repr__(self) -> str:
        match self.data_type:
            case "full" | "diagonal":
                return str(self.data_type)
            case _:
                return f"{self.data_type}_{self.row_number}"

    @property
    def conv_type(self) -> torch.nn.Module:
        return torch.nn.Conv2d if self.data_type == "full" else torch.nn.Conv1d

    @property
    def batchnorm_type(self) -> torch.nn.Module:
        return (
            torch.nn.BatchNorm2d if self.data_type == "full" else torch.nn.BatchNorm1d
        )


@dataclass
class MatrixEncoderActivationType:
    activation_type: str
    activation_params: dict | None

    @property
    def activation_fn(self) -> torch.nn.Module:
        match self.activation_type:
            case "lrelu":
                return torch.nn.LeakyReLU(self.activation_params["lrelu_slope"])
            case "gelu":
                return torch.nn.GELU()

    def __repr__(self) -> str:
        match self.activation_type:
            case "lrelu":
                return f"lrelu_{self.activation_params['lrelu_slope']}"
            case "gelu":
                return "gelu"


class ResponseMatrixEncoder(torch.nn.Module):
    def __init__(
        self,
        convolution_channels: list[int],
        use_batchnorm: bool,  # noqa: FBT001
        dropout_probability: float | None,
        encoder_type: MatrixEncoderType,
        activation: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.dropout_probability = dropout_probability
        self.encoder_type = encoder_type
        self.layers = self._construct_layers(convolution_channels, activation)

    def _construct_layers(
        self, convolution_channels: list[int], activation: torch.nn.Module
    ) -> torch.nn.Sequential:
        layers = torch.nn.Sequential()
        conv_type = self.encoder_type.conv_type
        batchnorm_type = self.encoder_type.batchnorm_type

        for in_channels, out_channels in zip(
            [1, *convolution_channels],
            convolution_channels,
        ):
            layers.append(
                conv_type(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding="same",
                ),
            )
            layers.append(activation)
            if self.use_batchnorm:
                layers.append(batchnorm_type(num_features=out_channels))
            if self.dropout_probability is not None and self.dropout_probability > 0:
                layers.append(torch.nn.Dropout(p=self.dropout_probability))

        layers.append(torch.nn.Flatten())
        layers.append(
            torch.nn.LazyLinear(out_features=16),
        )
        return layers

    def forward(self, matrices: torch.Tensor) -> torch.Tensor:
        return self.layers(self._get_encoder_input(matrices))

    def _get_encoder_input(self, matrices: torch.Tensor) -> torch.Tensor:
        assert matrices.ndim == 3  # B x R x C  # noqa: PLR2004
        match self.encoder_type:
            case MatrixEncoderType(data_type="full"):
                encoder_input = matrices
            case MatrixEncoderType(data_type="diagonal"):
                encoder_input = torch.diagonal(matrices, dim1=-2, dim2=-1)
            case MatrixEncoderType(data_type="row"):
                encoder_input = matrices[:, self.encoder_type.row_number, :]
            case MatrixEncoderType(data_type="column"):
                encoder_input = matrices[:, self.encoder_type.row_number, :]

        return encoder_input.unsqueeze(1)  # add channel dimension


class PathwayGNNEncoder(torch.nn.Module):
    def __init__(
        self,
        num_graph_layers: int,
        pooling_layer: PoolingLayerEnum,
        activation: str,
        network_type: str,
    ):
        super().__init__()
        network_class = {
            "graphsage": torch_geometric.nn.models.GraphSAGE,
            "gat": torch_geometric.nn.models.GAT,
            "gatv2": functools.partial(torch_geometric.nn.models.GAT, v2=True),
        }[network_type]
        self.gnn = network_class(
            in_channels=16,
            hidden_channels=16,
            num_layers=num_graph_layers,
            act=activation,
        )
        self.pooling_layer = pooling_layer.get_layer()
        self.linear = torch.nn.Linear(in_features=16, out_features=1)

    def forward(
        self,
        batch: torch_geometric.data.Batch,
        matrix_encodings: torch.Tensor,
    ) -> torch.Tensor:
        node_outputs = self.gnn(
            x=matrix_encodings,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )
        pool_output = self.pooling_layer(x=node_outputs, ptr=batch.ptr)
        return self.linear(pool_output)
