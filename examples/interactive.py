"""
Interactive example.
"""

import sys
from functools import partial
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import typer
from matplotlib.artist import Artist
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from differt2d.abc import LocEnum
from differt2d.geometry import DEFAULT_PATCH, FermatPath, ImagePath, MinPath, Point
from differt2d.logic import DEFAULT_ALPHA, DEFAULT_FUNCTION
from differt2d.scene import Scene
from differt2d.utils import P0, received_power

METHOD_TO_PATH_CLASS = {"image": ImagePath, "FPT": FermatPath, "MPT": MinPath}


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
        self.r_coef = 0.5
        self.path_cls = METHOD_TO_PATH_CLASS["image"]

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
        self.alpha_slider.setValue(self.alpha)
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
        self.patch_slider = LinearSlider(-1.0, 1.0)
        self.patch_slider.setValue(0.0)
        self.patch_label = QLabel("+0.00")

        grid.addWidget(QLabel("patch:"), 1, 1)
        grid.addWidget(self.patch_label, 1, 2)
        grid.addWidget(self.patch_slider, 1, 3)

        def set_patch(_):
            patch = self.patch_slider.value()
            self.patch = patch
            self.patch_label.setText(f"{patch:+.2f}")
            self.on_scene_change()

        self.patch_slider.valueChanged.connect(set_patch)

        self.min_order_spin_box = QSpinBox()
        self.min_order_spin_box.setMinimum(0)
        self.min_order_spin_box.setValue(self.min_order)

        grid.addWidget(QLabel("min. order:"), 2, 1)
        grid.addWidget(self.min_order_spin_box, 2, 3)

        def set_min_order(min_order):
            self.min_order = min_order
            self.on_scene_change()

        self.min_order_spin_box.valueChanged.connect(set_min_order)

        self.max_order_spin_box = QSpinBox()
        self.max_order_spin_box.setMinimum(0)
        self.max_order_spin_box.setValue(self.max_order)

        grid.addWidget(QLabel("max. order:"), 3, 1)
        grid.addWidget(self.max_order_spin_box, 3, 3)

        def set_max_order(max_order):
            self.max_order = max_order
            self.on_scene_change()

        self.max_order_spin_box.valueChanged.connect(set_max_order)

        self.r_coef_slider = LinearSlider(0.0, 1.0)
        self.r_coef_slider.setValue(self.r_coef)
        self.r_coef_label = QLabel("0.50")

        grid.addWidget(QLabel("refl. coef.:"), 4, 1)
        grid.addWidget(self.r_coef_label, 4, 2)
        grid.addWidget(self.r_coef_slider, 4, 3)

        def set_r_coef(_):
            r_coef = self.r_coef_slider.value()
            self.r_coef = r_coef
            self.r_coef_label.setText(f"{r_coef:.2f}")
            self.on_scene_change()

        self.r_coef_slider.valueChanged.connect(set_r_coef)

        self.path_cls_combo_box = QComboBox()
        self.path_cls_combo_box.addItems(METHOD_TO_PATH_CLASS.keys())
        self.path_cls_combo_box.setCurrentText(function)

        grid.addWidget(QLabel("method:"), 5, 1)
        grid.addWidget(self.path_cls_combo_box, 5, 3)

        def set_path_cls(method):
            self.path_cls = METHOD_TO_PATH_CLASS[method]
            self.on_scene_change()

        self.path_cls_combo_box.currentTextChanged.connect(set_path_cls)

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

        cm = self.coverage_map = self.ax.pcolormesh(
            self.X,
            self.Y,
            jnp.zeros_like(self.X),
            zorder=-2,
            vmin=-50,
            vmax=5,
        )
        cbar = self.fig.colorbar(cm, ax=self.ax)
        cbar.ax.set_ylabel("Power (dB)")

        scene.plot(
            ax=self.ax,
            annotate=False,
            emitters_kwargs=dict(picker=True),
            receivers_kwargs=dict(picker=True),
        )

        self.view.mpl_connect("pick_event", self.on_pick_event)
        self.view.mpl_connect("motion_notify_event", self.on_motion_notify_event)

        self.picked = None

        self.path_artists: List[Artist] = []

        def f(rx_coords):
            receivers = self.scene.receivers
            self.scene.receivers = {"rx": Point(point=rx_coords)}

            total_power = self.scene.accumulate_over_paths(
                fun=received_power,
                fun_kwargs=dict(r_coef=self.r_coef),
                reduce=True,
                min_order=self.min_order,
                max_order=self.max_order,
                path_cls=self.path_cls,
            )

            self.scene.receivers = receivers

            return total_power

        self.f_and_df = jax.value_and_grad(f)

        self.ax.autoscale(False, axis="x")
        self.ax.autoscale(False, axis="y")

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
        for artist in self.path_artists:
            artist.remove()

        self.path_artists = []

        P = self.scene.accumulate_on_receivers_grid_over_paths(
            self.X,
            self.Y,
            fun=received_power,
            fun_kwargs=dict(r_coef=self.r_coef),
            reduce=True,
            min_order=self.min_order,
            max_order=self.max_order,
            patch=self.patch,
            approx=self.approx,
            alpha=self.alpha,
            function=self.function,
            path_cls=self.path_cls,
        )

        PdB = 10.0 * jnp.log10(P / P0)

        self.coverage_map.set_array(PdB)

        for _, _, valid, path, _ in self.scene.all_paths(
            min_order=self.min_order,
            max_order=self.max_order,
            patch=self.patch,
            approx=self.approx,
            alpha=self.alpha,
            function=self.function,
            path_cls=self.path_cls,
        ):
            self.path_artists.extend(path.plot(self.ax, zorder=-1, alpha=float(valid)))

        if self.picked and False:
            _, point = self.picked

            x, y = rx_coords = point.point

            p, dp = self.f_and_df(rx_coords)

            ndp = jnp.linalg.norm(dp)

            if ndp > 0:
                dp = dp / ndp

            self.path_artists.extend(self.ax.quiver([x], [y], [dp[0]], [dp[1]]))

        self.view.draw()


def main(
    scene_name: Scene.SceneName = Scene.SceneName.basic_scene,
    file: Optional[Path] = None,
    resolution: int = 150,
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
