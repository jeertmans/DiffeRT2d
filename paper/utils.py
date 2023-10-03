from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def setup_fig_for_paper(
    fig: Figure,
    columns: int = 1,
    column_width: float = 252.0,
    height_to_width_ratio: Optional[float] = None,
) -> None:
    """
    Setup a figure to be nicely embedded in the IEEE paper.

    :param fig: The figure to setup.
    :param columns: The number of columns on which the figure should span.
    :param column_width: The width of one column (in pixels), defaults to the column
        width specified by the IEEE template.
    :param height_to_width_ratio: If set, the figure's height will be set accordingly,
        proportional to its width.
    """
    mpl.use("pgf")
    plt.rcParams.update(
        {
            "font.family": "serif",  # use serif/main font for text elements
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage[utf8x]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{cmbright}",
                ]
            ),
        }
    )

    px_to_inches = 0.0138889
    figwidth = px_to_inches * column_width * columns
    fig.set_figwidth(figwidth)

    if height_to_width_ratio:
        fig.set_figheight(height_to_width_ratio * figwidth)
