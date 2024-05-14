from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

mpl.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 10,
        "text.usetex": True,
        "pgf.rcfonts": False,
        "image.cmap": "plasma",
    }
)


def create_fig_for_paper(
    *args: Any,
    columns: int = 1,
    column_width: float = 252.0,
    height_to_width_ratio: Optional[float] = None,
    **kwargs,
) -> tuple[Figure, Any]:
    """
    Creates a new figure (and subplots), with a nice setup for IEEE papers.

    :param fig: The figure to setup.
    :param columns: The number of columns on which the figure should
        span.
    :param column_width: The width of one column (in pixels), defaults
        to the column width specified by the IEEE template.
    :param height_to_width_ratio: If set, the figure's height will be
        set accordingly, proportional to its width.
    """
    fig, axes = plt.subplots(*args, **kwargs)

    px_to_inches = 0.0138889
    figwidth = px_to_inches * column_width * columns
    fig.set_figwidth(figwidth)

    if height_to_width_ratio:
        fig.set_figheight(height_to_width_ratio * figwidth)

    return fig, axes
