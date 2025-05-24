import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mesa.visualization import make_space_component

from agent import Ant


def diffuse(field: np.ndarray, rate: float) -> np.ndarray:
    """
    One diffusion step with toroidal boundaries.

    Parameters
    ----------
    field : 2-D array of floats
    rate  : fraction [0, 1] that flows out of each cell this tick
            (0.25 is a good starting point; >0.5 will start to smear fast)

    Returns
    -------
    np.ndarray  (same shape) – diffused field
    """
    # Sum of the 8 neighbours (Moore neighbourhood)
    nbr_sum = (
            np.roll(field,  1, 0) + np.roll(field, -1, 0) +
            np.roll(field,  1, 1) + np.roll(field, -1, 1) +
            np.roll(np.roll(field,  1, 0),  1, 1) +  # NW
            np.roll(np.roll(field,  1, 0), -1, 1) +  # NE
            np.roll(np.roll(field, -1, 0),  1, 1) +  # SW
            np.roll(np.roll(field, -1, 0), -1, 1)    # SE
    )

    # Average of neighbours
    nbr_mean = nbr_sum / 8.0

    # Linear mix: keep (1-rate) of original + rate * neighbour average
    return field * (1.0 - rate) + nbr_mean * rate


def ant_portrayal(agent):
    if not isinstance(agent, Ant):
        return None  # ignore patches (we draw them via background heat‑maps)
    return {
        "color": "tab:orange" if agent.carrying else "tab:red",
        "size": 3,  # px marker diameter
        "marker": "o",
        "zorder": 2,
    }


def make_space_graph():
    """
    Factory that returns a SpaceMatplotlib component. Inside we
    build a post_process function that uses the model
    instance passed by the dashboard at run-time for plotting.
    """

    def MakeSpaceMatplotlib(model):
        def _heatmaps(ax, _model=model):
            """
            Draw pheromone, food and nest with transparent zeros.
            """

            def transparent_cmap(base_cmap, n=256):
                cmap = plt.get_cmap(base_cmap, n)
                colors = cmap(np.linspace(0, 1, n))
                colors[0, -1] = 0.0
                return mpl.colors.ListedColormap(colors)

            pher_cmap = transparent_cmap("Greens")
            food_cmap = transparent_cmap("Blues")
            nest_cmap = transparent_cmap("Purples")

            ax.set_xlim(0, _model.space.x_max)
            ax.set_ylim(0, _model.space.y_max)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_facecolor("white")  # explicit white background

            def im(arr, cmap, vmax):
                masked = np.ma.masked_where(arr == 0, arr)
                ax.imshow(
                    masked.T,
                    cmap=cmap,
                    alpha=1.0,
                    origin="lower",
                    vmin=0,
                    vmax=vmax,
                    interpolation="nearest",
                    zorder=0,
                )

            im(_model.pheromone, pher_cmap, vmax=20)
            im(_model.food, food_cmap, vmax=2)
            im(_model.nest, nest_cmap, vmax=1)

        Space = make_space_component(
            agent_portrayal=ant_portrayal,
            post_process=_heatmaps,
            figsize=(40, 40),
        )

        return Space(model)
    return MakeSpaceMatplotlib
