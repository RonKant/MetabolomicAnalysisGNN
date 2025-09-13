from collections.abc import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric

from src.data_processing.synthetic_data_generation import SyntheticSampleGenerator


def plot_sample(
    sample: torch_geometric.data.Data,
    use_original_matrices: bool = False,  # noqa: FBT001, FBT002
    num_rows: int = 1,
) -> None:
    sample_matrices = (
        SyntheticSampleGenerator._extract_sample_matrix_bank(  # noqa: SLF001
            sample,
            use_original_matrices=use_original_matrices,
        ).query("~is_empty")
    )

    num_cols = int(np.ceil(len(sample_matrices) / num_rows))

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, 3 * num_rows),
    )
    for ax in axes.reshape(-1):
        ax.set_axis_off()

    for ax, matrix, compound_name in zip(
        axes.reshape(-1),
        sample_matrices.matrix,
        sample_matrices.compound_name,
    ):
        matrix_transformed = torch.log10(torch.clip(matrix, min=10**3)) - 3
        ax.imshow(matrix_transformed, origin="lower")
        ax.set_title(compound_name)

    plt.tight_layout()


def plot_sample_graph(  # noqa: ANN201, PLR0913
    sample: torch_geometric.data.Data,
    normalize_values: bool = True,  # noqa: FBT001, FBT002
    figsize: tuple[int, int] | None = None,
    node_titles: Sequence[str] | None = None,
    pos=None,  # noqa: ANN001
    ax=None,  # noqa: ANN001
    v_minmax: tuple[int, int] | None = None,
):
    if v_minmax is None:
        vmax = sample.compound_mats.max().item()
        vmin = sample.compound_mats.min().item()
    else:
        vmin, vmax = v_minmax

    if node_titles is None:
        node_titles = [""] * len(sample.compound_mats)

    nx_graph = torch_geometric.utils.convert.to_networkx(data=sample)
    if pos is None:
        pos = nx.spring_layout(nx_graph, k=0.1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
    nx.draw(nx_graph, ax=ax, pos=pos, arrows=False)

    transform_figure = ax.transData.transform
    transform_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
    icon_center = icon_size / 2.0

    # # Add the respective image to each node
    for n, mat, title in zip(
        nx_graph.nodes,
        sample.compound_mats,
        node_titles,
        strict=True,
    ):
        if (mat > 0).any():
            xf, yf = transform_figure(pos[n])
            xa, ya = transform_axes((xf, yf))
            # get overlapped axes and plot icon
            a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
            if normalize_values:
                a.imshow(
                    mat.detach().cpu().numpy(),
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                )
            else:
                a.imshow(mat.detach().cpu().numpy(), origin="lower")
            a.set_title(title)
            a.axis("off")

    return pos, (vmin, vmax)
