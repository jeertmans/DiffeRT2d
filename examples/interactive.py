import sys

from functools import partial
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer

from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from differt2d.abc import LocEnum
from differt2d.logic import enable_approx
from differt2d.scene import Scene
from differt2d.utils import flatten

EPS = jnp.finfo(float).eps


@partial(jax.jit, inline=True)
def power(path, path_candidate, objects):
    l1 = path.length()
    l2 = l1 * l1
    c = 0.5  # Power attenuation from one wall
    n = len(path_candidate) - 2  # Number of walls
    a = c**n
    p = a / (EPS + l2)

    return (p - 1.0) ** 2


class PlotWidget(QWidget):
    def __init__(self, scene: Scene, min_order: int, max_order: int, parent=None):
        super().__init__(parent)

        self.scene = scene
        self.min_order = min_order
        self.max_order = max_order

        # -- Create widgets --

        # Matplotlib figures
        self.fig = Figure(figsize=(10, 10), tight_layout=True)
        self.view = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot()

        # Toolbar above the figure
        self.toolbar = NavigationToolbar2QT(self.view, self)

        # Figures and toolbar

        main_layout = QVBoxLayout()

        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.view)

        self.setLayout(main_layout)

        scene.plot(
            ax=self.ax,
            annotate=False,
            emitters_kwargs=dict(picker=True),
            receivers_kwargs=dict(picker=True),
        )

        self.view.mpl_connect("pick_event", self.on_pick_event)
        self.view.mpl_connect("motion_notify_event", self.on_motion_notify_event)

        self.picked = None

        self.path_artists = []

        self.on_scene_change()

    def on_pick_event(self, event):
        if self.picked:
            self.picked = None
        else:
            artist = event.artist
            coords = jnp.array(artist.get_xydata())
            tx, dist_tx = self.scene.get_closest_emitter(coords)
            rx, dist_rx = self.scene.get_closest_receiver(coords)

            if dist_tx < dist_rx:
                self.picked = (event.artist, tx)
            else:
                self.picked = (event.artist, rx)

    def on_motion_notify_event(self, event):
        if self.picked:
            artist, point = self.picked
            point.point = jnp.array([event.xdata, event.ydata])
            artist.set_xdata([event.xdata])
            artist.set_ydata([event.ydata])
            self.on_scene_change()

    def on_scene_change(self):
        """
        The scene has changed, we must update the plot.
        """
        for artist in flatten(self.path_artists):
            artist.remove()

        self.path_artists = []

        for paths in self.scene.all_paths(
            min_order=self.min_order, max_order=self.max_order
        ).values():
            for path in paths:
                self.path_artists.append(path.plot(self.ax, zorder=-1))

        self.view.draw()


def main(
    scene_name: Scene.SceneName = Scene.SceneName.basic_scene,
    file: Optional[Path] = None,
    min_order: int = 0,
    max_order: int = 1,
    approx: bool = True,
    tx_loc: LocEnum = LocEnum.C,
    rx_loc: LocEnum = LocEnum.S,
):
    if file:
        scene = Scene.from_geojson(
            file.read_text(), tx_loc=tx_loc.value, rx_loc=rx_loc.value
        )
    else:
        scene = Scene.from_scene_name(scene_name)

    with enable_approx(approx):
        app = QApplication(sys.argv)
        plot_widget = PlotWidget(scene=scene, min_order=min_order, max_order=max_order)
        plot_widget.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    typer.run(main)
