from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .config import (
    SimulationConfig,
    fft2_centered,
    make_hologram,
    propagate_angular_spectrum,
)


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


def plot_hologram_recording(
    object_field: np.ndarray,
    reference_field: np.ndarray,
    config: SimulationConfig,
) -> Figure:
    """Plot the recorded hologram, complex fields, and its centered spectrum."""
    hologram = make_hologram(object_field, reference_field)
    total_field = np.asarray(object_field) + np.asarray(reference_field)
    spectrum = np.abs(fft2_centered(hologram))
    spectrum_display = np.log1p(spectrum)

    x, y = config.coordinate_grid()
    spatial_extent = (
        x.min() * 1e3,
        x.max() * 1e3,
        y.min() * 1e3,
        y.max() * 1e3,
    )
    fx, fy = config.frequency_grid()
    frequency_extent = (
        fx.min() * 1e-3,
        fx.max() * 1e-3,
        fy.min() * 1e-3,
        fy.max() * 1e-3,
    )
    amplitude_fields = (object_field, reference_field, total_field)
    amplitude_titles = ("Object amplitude", "Reference amplitude", "Sensor field amplitude")
    phase_titles = ("Object phase", "Reference phase", "Sensor field phase")
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)

    for column, (field, amplitude_title, phase_title) in enumerate(
        zip(amplitude_fields, amplitude_titles, phase_titles)
    ):
        amplitude = np.abs(field)
        scale = float(np.max(amplitude))
        axes[0, column].imshow(
            amplitude / scale if scale > 0.0 else amplitude,
            extent=spatial_extent,
            origin="lower",
            vmin=0.0,
            cmap="inferno",
        )
        axes[0, column].set_title(amplitude_title)
        axes[1, column].imshow(
            np.angle(field),
            extent=spatial_extent,
            origin="lower",
            vmin=-np.pi,
            vmax=np.pi,
            cmap="twilight",
        )
        axes[1, column].set_title(phase_title)
        for row in range(2):
            axes[row, column].set_xlabel("x (mm)")
            axes[row, column].set_ylabel("y (mm)")

    intensity_scale = float(np.max(hologram))
    axes[0, 3].imshow(
        hologram / intensity_scale if intensity_scale > 0.0 else hologram,
        extent=spatial_extent,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="inferno",
    )
    axes[0, 3].set_title("Normalized hologram intensity")
    axes[0, 3].set_xlabel("x (mm)")
    axes[0, 3].set_ylabel("y (mm)")
    axes[1, 3].imshow(
        spectrum_display,
        extent=frequency_extent,
        origin="lower",
        cmap="viridis",
    )
    axes[1, 3].plot(
        [
            -config.reference_frequency_x * 1e-3,
            config.reference_frequency_x * 1e-3,
        ],
        [
            config.reference_frequency_y * 1e-3,
            -config.reference_frequency_y * 1e-3,
        ],
        linestyle="none",
        marker="x",
        color="red",
    )
    axes[1, 3].set_title("log(1 + |FFT(hologram)|)")
    axes[1, 3].set_xlabel("fx (cycles/mm)")
    axes[1, 3].set_ylabel("fy (cycles/mm)")
    figure.suptitle("Off-axis hologram recording")
    return figure


def plot_reconstruction(field: np.ndarray, config: SimulationConfig) -> Figure:
    """Plot peak-normalized amplitude and wrapped phase of an object-plane field."""
    field_array = np.asarray(field)
    expected_shape = (config.ny, config.nx)
    if (
        field_array.ndim != 2
        or field_array.shape != expected_shape
        or not np.issubdtype(field_array.dtype, np.number)
        or not np.all(np.isfinite(field_array))
    ):
        raise ValueError(f"field must be a finite numeric array with shape {expected_shape}.")
    amplitude = np.abs(field_array)
    amplitude_scale = float(np.max(amplitude))
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0.0:
        raise ValueError("field must have a finite, non-zero amplitude.")

    x, y = config.coordinate_grid()
    extent = (
        x.min() * 1e3,
        x.max() * 1e3,
        y.min() * 1e3,
        y.max() * 1e3,
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].imshow(
        amplitude / amplitude_scale,
        extent=extent,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
        cmap="inferno",
    )
    axes[0].set_title("Reconstructed amplitude (normalized to peak)")
    axes[1].imshow(
        np.angle(field_array),
        extent=extent,
        origin="upper",
        vmin=-np.pi,
        vmax=np.pi,
        cmap="twilight",
    )
    axes[1].set_title("Reconstructed wrapped phase [-pi, pi] (rad)")
    for axis in axes:
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    figure.suptitle("Digital hologram reconstruction")
    return figure
