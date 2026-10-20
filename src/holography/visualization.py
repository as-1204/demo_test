from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .config import SimulationConfig, propagate_angular_spectrum


def plot_propagation(
    field: np.ndarray,
    config: SimulationConfig,
    distances: np.ndarray | None = None,
) -> Figure:
    """Plot normalized intensity and phase at evenly spaced propagation planes."""
    if distances is None:
        distances = np.linspace(0.0, config.propagation_distance, 5)
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1 or distances.size == 0 or not np.all(np.isfinite(distances)):
        raise ValueError("distances must be a non-empty 1D array of finite values.")

    x, y = config.coordinate_grid()
    extent = (
        x.min() * 1e3,
        x.max() * 1e3,
        y.min() * 1e3,
        y.max() * 1e3,
    )
    object_intensity = np.abs(field) ** 2
    intensity_scale = float(np.max(object_intensity))
    if not np.isfinite(intensity_scale) or intensity_scale <= 0.0:
        raise ValueError("field must have a finite, non-zero intensity.")

    figure, axes = plt.subplots(
        2,
        distances.size,
        figsize=(3.2 * distances.size, 6.0),
        squeeze=False,
        constrained_layout=True,
    )
    for column, distance in enumerate(distances):
        propagated = propagate_angular_spectrum(field, config, distance=float(distance))
        normalized_intensity = np.abs(propagated) ** 2 / intensity_scale
        phase = np.angle(propagated)
        axes[0, column].imshow(
            normalized_intensity,
            extent=extent,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="inferno",
        )
        axes[0, column].set_title(f"z = {distance * 100:.1f} cm")
        axes[1, column].imshow(
            phase,
            extent=extent,
            origin="lower",
            vmin=-np.pi,
            vmax=np.pi,
            cmap="twilight",
        )
        axes[1, column].set_title(f"z = {distance * 100:.1f} cm")
        for row in range(2):
            axes[row, column].set_xlabel("x (mm)")
            axes[row, column].set_ylabel("y (mm)")

    axes[0, 0].set_ylabel("Intensity\n y (mm)")
    axes[1, 0].set_ylabel("Phase (rad)\n y (mm)")
    figure.suptitle("Angular-spectrum propagation")
    return figure
