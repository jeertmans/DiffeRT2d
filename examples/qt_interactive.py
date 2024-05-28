"""
Interactive example with Qt interface
=====================================

This example provides an interactive experience to most of the features
proposed by this module.

.. warning::
    As of writing this, the ``FPT`` and ``MPT`` path methods
    will not show the expected coverage (which ``image`` usually
    does). This is mainly an issue caused by the use of `only`
    100 steps for the minimization, which may not be necessary
    to properly converge.

    Also, you will observe that the coverage map changes a lot between two
    position, this is because the random key used to initialize
    the minimization process changes on each update.

Requirements
------------

This example requires additional Python modules, namely ``qtpy`` and
Qt5 bindings (we recommend using PySide6).
You can install the necessary modules with ``pip install qtpy pyside6``.

Usage
-----

Running this example is as simple as ``python examples/qt-interactive.py``.

However, you can specify a variety of parameters directly when
calling the CLI. See ``python examples/qt-interactive.py --help``.

Getting help
------------

Using the graphical interface should be straightforward.

On top of that, a lot of widgets display contextual information
if you hover over them.
"""

# %%
# Imports
# -------
#
# First, we need to import the necessary modules.
# Note that most imports are only required to create the
# graphical interface, not actually doing the ray tracing.

from argparse import ArgumentParser, FileType
from functools import partial
from typing import Literal, get_args

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from matplotlib.artist import Artist
from matplotlib.backend_bases import MouseEvent, PickEvent
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,  # type: ignore[reportAttributeAccessIssue]
    NavigationToolbar2QT,  # type: ignore[reportPrivateImportUsage]
)
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
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

from differt2d.abc import Loc
from differt2d.defaults import DEFAULT_ALPHA, DEFAULT_PATCH
from differt2d.geometry import FermatPath, ImagePath, MinPath, Point
from differt2d.logic import hard_sigmoid, sigmoid
from differt2d.scene import Scene, SceneName
from differt2d.utils import P0, received_power

METHOD_TO_PATH_CLASS = {"image": ImagePath, "FPT": FermatPath, "MPT": MinPath}

# %%
# GUI classes
# -----------
#
# The following defines all GUI-related classes and methods,
# needed to display the application.


class CustomSlider(QSlider):
    def __init__(
        self,
        start: float = 0,
        stop: float = 99,
        num: int = 100,
        base: float = 10.0,
        scale: Literal["linear", "log", "geom"] = "linear",
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

    def value(self) -> float:  # type: ignore[reportIncompatibleMethodOverride]
        index = super().value()
        return float(self.values[index])

    def maximum(self):  # type: ignore[reportIncompatibleMethodOverride]
        return float(self.values[-1])

    def minimum(self):  # type: ignore[reportIncompatibleMethodOverride]
        return float(self.values[0])

    def setValue(self, value: float) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        index = jnp.abs(self.values - value).argmin()
        super().setValue(int(index))


LinearSlider = partial(CustomSlider, scale="linear")
LogSlider = partial(CustomSlider, scale="log")
GeomSlider = partial(CustomSlider, scale="geom")


class PlotWidget(QWidget):
    def __init__(
        self,
        scene: Scene,
        resolution: int,
        seed: int,
        parent=None,
    ):
        super().__init__(parent)

        self.scene = scene
        self.min_order = 0
        self.max_order = 1
        self.patch = DEFAULT_PATCH
        self.approx = False
        self.alpha = DEFAULT_ALPHA
        self.function = hard_sigmoid
        self.r_coef = 0.5
        self.path_cls = METHOD_TO_PATH_CLASS["image"]
        self.key = jax.random.PRNGKey(seed)

        assert (
            len(scene.transmitters) == 1
        ), "This simulation only supports one transmitter"
        assert len(scene.receivers) == 1, "This simulation only supports one receiver"

        # -- Create widgets --

        # Fix ToolTip with dark theme issue

        self.setStyleSheet(
            """
            QToolTip {
                background-color: #ea5626;
            }"""
        )

        # Approx. parameters
        approx_box = QGroupBox("Enable approx.")
        approx_box.setCheckable(True)
        approx_box.setChecked(False)
        approx_box.setToolTip(
            "Click to enable/disable approximation. "
            "When enabled, you can specify further parameters below."
        )

        def set_approx(approx):
            self.approx = approx
            self.on_scene_change()

        approx_box.toggled.connect(set_approx)

        grid = QGridLayout()
        approx_box.setLayout(grid)
        self.alpha_slider = LogSlider(0, 3)
        self.alpha_slider.setValue(self.alpha)
        self.alpha_slider.setToolTip(
            "The alpha value, as used by the activation function"
        )
        self.alpha_label = QLabel("1.000e+00")

        grid.addWidget(QLabel("alpha:"), 1, 1)
        grid.addWidget(self.alpha_label, 1, 2)
        grid.addWidget(self.alpha_slider, 1, 3)

        def set_alpha(_) -> None:
            alpha = self.alpha_slider.value()
            self.alpha = alpha
            self.alpha_label.setText(f"{alpha:.3e}")
            self.on_scene_change()

        self.alpha_slider.valueChanged.connect(set_alpha)

        self.function_combo_box = QComboBox()
        self.function_combo_box.addItems(["sigmoid", "hard_sigmoid"])
        self.function_combo_box.setToolTip(" The activation function")
        self.function_combo_box.setCurrentText("hard_sigmoid")

        grid.addWidget(QLabel("activation function:"), 2, 1)
        grid.addWidget(self.function_combo_box, 2, 3)

        def set_function(function):
            self.function = {
                "sigmoid": sigmoid,
                "hard_sigmoid": hard_sigmoid,
            }[function]
            self.on_scene_change()

        self.function_combo_box.currentTextChanged.connect(set_function)

        # General settings
        settings_box = QGroupBox("General Setting")

        grid = QGridLayout()
        settings_box.setLayout(grid)
        self.patch_slider = LinearSlider(-1.0, 1.0)
        self.patch_slider.setValue(0.0)
        self.patch_slider.setToolTip(
            "The patch value used to virtually increase/decrease objects' "
            "size when checking for intersection"
        )
        self.patch_label = QLabel("+0.00")

        grid.addWidget(QLabel("patch:"), 1, 1)
        grid.addWidget(self.patch_label, 1, 2)
        grid.addWidget(self.patch_slider, 1, 3)

        def set_patch(_) -> None:
            patch = self.patch_slider.value()
            self.patch = patch
            self.patch_label.setText(f"{patch:+.2f}")
            self.on_scene_change()

        self.patch_slider.valueChanged.connect(set_patch)

        self.min_order_spin_box = QSpinBox()
        self.min_order_spin_box.setMinimum(0)
        self.min_order_spin_box.setToolTip(
            "The minimum interaction order, 0 is line of sight."
        )
        self.min_order_spin_box.setValue(self.min_order)

        grid.addWidget(QLabel("min. order:"), 2, 1)
        grid.addWidget(self.min_order_spin_box, 2, 3)

        def set_min_order(min_order: int) -> None:
            self.min_order = min_order
            self.on_scene_change()

        self.min_order_spin_box.valueChanged.connect(set_min_order)

        self.max_order_spin_box = QSpinBox()
        self.max_order_spin_box.setMinimum(0)
        self.max_order_spin_box.setToolTip(
            "The maximum interaction order, 1 is one reflection max."
        )
        self.max_order_spin_box.setValue(self.max_order)

        grid.addWidget(QLabel("max. order:"), 3, 1)
        grid.addWidget(self.max_order_spin_box, 3, 3)

        def set_max_order(max_order: int) -> None:
            self.max_order = max_order
            self.on_scene_change()

        self.max_order_spin_box.valueChanged.connect(set_max_order)

        self.r_coef_slider = LinearSlider(0.0, 1.0)
        self.r_coef_slider.setValue(self.r_coef)
        self.r_coef_slider.setToolTip(
            "The reflection coefficient, expressed as a real value between 0 and 1."
        )
        self.r_coef_label = QLabel("0.50")

        grid.addWidget(QLabel("refl. coef.:"), 4, 1)
        grid.addWidget(self.r_coef_label, 4, 2)
        grid.addWidget(self.r_coef_slider, 4, 3)

        def set_r_coef(_) -> None:
            r_coef = self.r_coef_slider.value()
            self.r_coef = r_coef
            self.r_coef_label.setText(f"{r_coef:.2f}")
            self.on_scene_change()

        self.r_coef_slider.valueChanged.connect(set_r_coef)

        self.path_cls_combo_box = QComboBox()
        self.path_cls_combo_box.addItems(METHOD_TO_PATH_CLASS.keys())  # type: ignore
        self.path_cls_combo_box.setToolTip(
            "The path method that is used to determine each ray path. "
            "Note that each next method is much slower than the previous, "
            "but can simulate more complex interaction types, "
            "like diffraction, not implemented at the moment."
        )
        self.path_cls_combo_box.setCurrentText("image")

        grid.addWidget(QLabel("method:"), 5, 1)
        grid.addWidget(self.path_cls_combo_box, 5, 3)

        def set_path_cls(method: str) -> None:
            self.path_cls = METHOD_TO_PATH_CLASS[method]
            self.on_scene_change()

        self.path_cls_combo_box.currentTextChanged.connect(set_path_cls)

        # Matplotlib figures
        self.fig = Figure(figsize=(10, 10), tight_layout=True)
        self.view = FigureCanvas(self.fig)
        self.view.setMinimumHeight(200)
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
            transmitters_kwargs=dict(picker=True),
            receivers_kwargs=dict(picker=True),
        )

        self.view.mpl_connect("pick_event", self.on_pick_event)
        self.view.mpl_connect("motion_notify_event", self.on_motion_notify_event)

        self.picked = None

        self.path_artists: list[Artist] = []

        def f(rx_coords: Float[Array, "2"]) -> Float[Array, " "]:
            scene = self.scene.with_transmitters(rx=Point(xy=rx_coords))

            self.key, key = jax.random.split(self.key, 2)

            total_power = scene.accumulate_over_paths(
                fun=partial(received_power, r_coef=self.r_coef),
                reduce_all=True,
                min_order=self.min_order,
                max_order=self.max_order,
                path_cls=self.path_cls,
                key=key,
            )

            return total_power  # type: ignore

        self.f_and_df = jax.value_and_grad(f)

        xlim, ylim = self.scene.bounding_box().T
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.autoscale(False, axis="x")
        self.ax.autoscale(False, axis="y")

        self.on_scene_change()

    def on_pick_event(self, event: PickEvent) -> None:
        if self.picked:
            self.picked = None
        else:
            artist = event.artist
            coords = jnp.array(artist.get_xydata()).reshape(-1)  # type: ignore
            tx, dist_tx = self.scene.get_closest_transmitter(coords)
            rx, dist_rx = self.scene.get_closest_receiver(coords)

            if dist_tx < dist_rx:
                self.picked = (event.artist, tx)
                self.picked_tx = True
            else:
                self.picked = (event.artist, rx)
                self.picked_tx = False

    def on_motion_notify_event(self, event: MouseEvent) -> None:
        if self.picked and event.xdata is not None and event.ydata is not None:
            artist, point_name = self.picked
            if self.picked_tx:
                self.scene = self.scene.update_transmitters(
                    **{point_name: Point(xy=jnp.array([event.xdata, event.ydata]))}
                )
            else:
                self.scene = self.scene.update_receivers(
                    **{point_name: Point(xy=jnp.array([event.xdata, event.ydata]))}
                )
            artist.set_xdata([event.xdata])  # type: ignore
            artist.set_ydata([event.ydata])  # type: ignore
            self.on_scene_change()

    def on_scene_change(self) -> None:
        for artist in self.path_artists:
            artist.remove()

        self.path_artists = []

        self.key, key_acc, key_all = jax.random.split(self.key, 3)

        P: Float[Array, "n n"] = self.scene.accumulate_on_receivers_grid_over_paths(
            self.X,
            self.Y,
            fun=partial(received_power, r_coef=self.r_coef),
            reduce_all=True,
            min_order=self.min_order,
            max_order=self.max_order,
            patch=self.patch,
            approx=self.approx,
            alpha=self.alpha,
            function=self.function,
            path_cls=self.path_cls,
            key=key_acc,
        )  # type: ignore

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
            key=key_all,
        ):
            self.path_artists.extend(path.plot(self.ax, zorder=-1, alpha=float(valid)))

        if self.picked and False:
            _, point = self.picked

            x, y = rx_coords = point.xy

            p, dp = self.f_and_df(rx_coords)

            ndp = jnp.linalg.norm(dp)

            if ndp > 0:
                dp = dp / ndp

            self.path_artists.extend(self.ax.quiver([x], [y], [dp[0]], [dp[1]]))

        self.view.draw()


# %%
# CLI options
# -----------
#
# This part is not very interesting, and uses the builtin :mod:`argparse`
# module to create a set of command-line options and parse them.


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="interactive-example",
        description="DiffeRT2d's interactive example.",
        epilog="This example shows most of the features available in this "
        "Python module. Feel free to modify the various parameters using "
        "the sliders and other widgets.",
    )
    parser.add_argument(
        "--scene",
        metavar="NAME",
        default="basic_scene",
        choices=get_args(SceneName),
        help="select scene by name (default: basic_scene, "
        f"allowed: {', '.join(get_args(SceneName))})",
        dest="scene_name",
    )
    parser.add_argument(
        "--resolution",
        metavar="INT",
        type=int,
        default=150,
        choices=range(1, 999999),
        help="set the grid resolution (default: 150, allowed: 1 to 999999 excl.)",
    )
    parser.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        default=1234,
        choices=range(1, 999999),
        help="set the grid resolution (default: 1234, allowed: 1 to 999999 excl.)",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        type=FileType("r"),
        default=None,
        help="if present, read the scene from the specified file path",
    )
    parser.add_argument(
        "--tx-loc",
        metavar="Loc",
        default="NW",
        choices=get_args(Loc),
        help="when file is set, set the transmitter location (default: NW, "
        f"allowed: {', '.join(get_args(Loc))})",
    )
    parser.add_argument(
        "--rx-loc",
        metavar="Loc",
        default="SE",
        choices=get_args(Loc),
        help="when file is set, set the receiver location (default: SE, "
        f"allowed: {', '.join(get_args(Loc))})",
    )
    args = parser.parse_args()

    if args.file:
        scene = Scene.from_geojson(
            args.file.read(), tx_loc=args.tx_loc, rx_loc=args.rx_loc
        )
    else:
        scene = Scene.from_scene_name(args.scene_name)

    app = QApplication.instance()

    if app is None:
        app = QApplication([])

    plot_widget = PlotWidget(
        scene=scene,
        resolution=args.resolution,
        seed=args.seed,
    )
    plot_widget.show()
    app.exec_()
