import sys
from copy import copy
from functools import partial
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import typer
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSlider, QComboBox, QApplication, QGridLayout, QGroupBox, QVBoxLayout, QWidget

from differt2d.abc import LocEnum
from differt2d.geometry import DEFAULT_PATCH, Point
from differt2d.logic import DEFAULT_ALPHA, DEFAULT_FUNCTION
from differt2d.scene import Scene
from differt2d.utils import flatten

EPS = jnp.finfo(float).eps

@jax.jit
def power(emitter, receiver, path, interacting_objects, P0=1.0, r_coef=0.5, epsilon=EPS):
    """
    Computes the power received from a given path.
    """
    r = path.length()
    n = len(path) - 2  # Number of walls
    return P0 * (r_coef**n) / (epsilon + r * r)


class CustomSlider(QSlider):
    def __init__(
        self,
        start=0,
        stop=99,
        num=100,
        base=10.0,
        scale="linear",
        orientation=Qt.Orientation.Horizontal,
    ):
        super().__init__()
        self.setMinimum(0)
        self.setMaximum(num - 1)
        self.setOrientation(orientation)

        if scale == "linear":
            self.values = jnp.linspace(start, stop, num=num)
        elif scale == "log":
            self.values = jnp.logspace(start, stop, num=num, base=base)
        elif scale == "geom":
            self.values = jnp.geomspace(start, stop, num=num)
        else:
            raise ValueError("Invalid scale: " + scale)

    def value(self):
        index = super().value()
        return self.values[index]

    def maximum(self):
        return self.values[-1]

    def minimum(self):
        return self.values[0]

    def setValue(self, value):
        index = jnp.abs(self.values - value).argmin()
        super().setValue(index)


LinearSlider = partial(CustomSlider, scale="linear")
LogSlider = partial(CustomSlider, scale="log")
GeomSlider = partial(CustomSlider, scale="geom")


class PlotWidget(QWidget):
    def __init__(
        self,
        scene: Scene,
        resolution: int,
        min_order: int,
        max_order: int,
        patch: float,
        approx: bool,
        alpha: float,
        function: str,
        parent=None,
    ):
        super().__init__(parent)

        self.scene = scene
        self.min_order = min_order
        self.max_order = max_order
        self.patch = patch
        self.approx = approx
        self.alpha = alpha
        self.function = function

        assert len(scene.emitters) == 1, "This simulation only supports one emitter"
        assert len(scene.receivers) == 1, "This simulation only supports one receiver"

        # -- Create widgets --

        # Approx. parameters
        approx_box = QGroupBox("Enable approx.")
        approx_box.setCheckable(True)
        approx_box.setChecked(approx)

        def set_approx(approx):
            self.approx = approx
            self.on_scene_change()

        approx_box.toggled.connect(set_approx)

        grid = QGridLayout()
        approx_box.setLayout(grid)
        self.alpha_slider = LogSlider(0, 3)
        self.alpha_label = QLabel("1.000e+00")

        grid.addWidget(QLabel("alpha:"), 1, 1)
        grid.addWidget(self.alpha_label, 1, 2)
        grid.addWidget(self.alpha_slider, 1, 3)

        def set_alpha(_):
            alpha = self.alpha_slider.value()
            self.alpha = alpha
            self.alpha_label.setText(f"{alpha:.3e}")
            self.on_scene_change()

        self.alpha_slider.valueChanged.connect(set_alpha)

        self.function_combo_box = QComboBox()
        self.function_combo_box.addItems(["sigmoid", "hard_sigmoid"])
        self.function_combo_box.setCurrentText(function)

        grid.addWidget(QLabel("activation function:"), 2, 1)
        grid.addWidget(self.function_combo_box, 2, 3)

        def set_function(function):
            self.function = function
            self.on_scene_change()

        self.function_combo_box.currentTextChanged.connect(set_function)

        # General settings
        settings_box = QGroupBox("General Setting")

        grid = QGridLayout()
        settings_box.setLayout(grid)
        self.patch_slider = LinearSlider(-1., 1.)
        self.patch_slider.setValue(0.)
        self.patch_label = QLabel("+0.00")

        grid.addWidget(QLabel("patch:"), 1, 1)
        grid.addWidget(self.patch_label, 1, 2)
        grid.addWidget(self.patch_slider, 1, 3)

        def set_patch(_):
            patch = self.patch_slider.value()
            self.patch = patch
            self.patch_label.setText(f"{patch:.2}")
            self.on_scene_change()

        self.patch_slider.valueChanged.connect(set_patch)

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
        main_layout.addWidget(approx_box)
        main_layout.addWidget(settings_box)

        self.setLayout(main_layout)

        self.X, self.Y = self.scene.grid(n=resolution)

        self.coverage_map = self.ax.pcolormesh(
            self.X, self.Y, self.X + self.Y, zorder=-2
        )

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

        def f(rx_coords):
            receivers = self.scene.receivers
            self.scene.receivers = {"rx": Point(point=rx_coords)}

            total_power = self.scene.accumulate_over_paths(
                fun=power,
                min_order=self.min_order, max_order=self.max_order
            )

            self.scene.receivers = receivers

            return total_power

        self.f_and_df = jax.value_and_grad(f)

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
        for artist in flatten(self.path_artists):
            artist.remove()

        self.path_artists = []

        Z = self.scene.accumulate_on_receivers_grid_over_paths(
            self.X, self.Y, fun=power,
            min_order=self.min_order,
            max_order=self.max_order,
            patch=self.patch,
            approx=self.approx,
            alpha=self.alpha,
            function=self.function,
        )

        self.coverage_map.set_array(jnp.log1p(Z))

        for _, _, path, _ in self.scene.all_valid_paths(
            min_order=self.min_order,
            max_order=self.max_order,
            patch=self.patch,
            approx=self.approx,
            alpha=self.alpha,
            function=self.function,
        ):
            self.path_artists.append(path.plot(self.ax, zorder=-1))

        if self.picked and False:
            _, point = self.picked

            x, y = rx_coords = point.point

            p, dp = self.f_and_df(rx_coords)

            ndp = jnp.linalg.norm(dp)

            if ndp > 0:
                dp = dp / ndp

            self.path_artists.append(self.ax.quiver([x], [y], [dp[0]], [dp[1]]))
            print(p, dp)

        self.view.draw()


def main(
    scene_name: Scene.SceneName = Scene.SceneName.basic_scene,
    file: Optional[Path] = None,
    resolution: int = 50,
    min_order: int = 0,
    max_order: int = 1,
    patch: float = DEFAULT_PATCH,
    approx: bool = True,
    alpha: float = DEFAULT_ALPHA,
    function: str = DEFAULT_FUNCTION,
    tx_loc: LocEnum = LocEnum.C,
    rx_loc: LocEnum = LocEnum.S,
):
    if file:
        scene = Scene.from_geojson(
            file.read_text(), tx_loc=tx_loc.value, rx_loc=rx_loc.value
        )
    else:
        scene = Scene.from_scene_name(scene_name)

    app = QApplication(sys.argv)
    plot_widget = PlotWidget(
        scene=scene,
        resolution=resolution,
        min_order=min_order,
        max_order=max_order,
        patch=patch,
        approx=approx,
        alpha=alpha,
        function=function,
    )
    plot_widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    typer.run(main)
