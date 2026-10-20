from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Minimal scalar-holography configuration aligned with the Phase 0 design notes."""

    wavelength: float = 532e-9
    pixel_pitch: float = 8e-6
    nx: int = 512
    ny: int = 512
    propagation_distance: float = 0.10
    reference_amplitude: float = 1.0
    reference_frequency_x: float = 20_000.0
    reference_frequency_y: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.wavelength) or self.wavelength <= 0.0:
            raise ValueError("wavelength must be finite and positive.")
        if not math.isfinite(self.pixel_pitch) or self.pixel_pitch <= 0.0:
            raise ValueError("pixel_pitch must be finite and positive.")
        if not math.isfinite(self.propagation_distance):
            raise ValueError("propagation_distance must be finite.")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive integers.")
        if not math.isfinite(self.reference_amplitude) or self.reference_amplitude < 0.0:
            raise ValueError("reference_amplitude must be finite and non-negative.")
        if not math.isfinite(self.reference_frequency_x) or not math.isfinite(self.reference_frequency_y):
            raise ValueError("reference frequencies must be finite.")

    @property
    def dx(self) -> float:
        return self.pixel_pitch

    @property
    def dy(self) -> float:
        return self.pixel_pitch

    def coordinate_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        return make_coordinate_grid(self.nx, self.ny, self.dx, self.dy)

    def frequency_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        return make_frequency_grid(self.nx, self.ny, self.dx, self.dy)

    def wave_number_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        return make_wave_number_grid(self.nx, self.ny, self.dx, self.dy)


def make_coordinate_grid(nx: int, ny: int, dx: float, dy: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(x, y)` coordinate grids for a rectangular field.

    The convention follows the roadmap:
      x[j] = (j - floor(Nx / 2)) * dx
      y[i] = (floor(Ny / 2) - i) * dy
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers.")
    if dx <= 0.0:
        raise ValueError("dx must be positive.")

    dy = dx if dy is None else dy
    if dy <= 0.0:
        raise ValueError("dy must be positive.")

    x = (np.arange(nx, dtype=float) - np.floor(nx / 2)) * dx
    y = (np.floor(ny / 2) - np.arange(ny, dtype=float)) * dy
    return np.meshgrid(x, y, indexing="xy")


def make_frequency_grid(nx: int, ny: int, dx: float, dy: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(fx, fy)` frequency grids in cycles/m using FFT shift conventions."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers.")
    if dx <= 0.0:
        raise ValueError("dx must be positive.")

    dy = dx if dy is None else dy
    if dy <= 0.0:
        raise ValueError("dy must be positive.")

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    return np.meshgrid(fx, fy, indexing="xy")


def make_wave_number_grid(nx: int, ny: int, dx: float, dy: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(kx, ky)` wave-number grids in rad/m."""
    fx, fy = make_frequency_grid(nx, ny, dx, dy)
    return 2.0 * np.pi * fx, 2.0 * np.pi * fy


def fft2_centered(field: np.ndarray) -> np.ndarray:
    """Forward FFT using the centered-field convention used in the roadmap."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def ifft2_centered(field: np.ndarray) -> np.ndarray:
    """Inverse FFT for the same centered-field convention."""
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field)))


def make_complex_field(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Combine non-negative amplitude and phase (radians) into a complex field."""
    raw_amplitude = np.asarray(amplitude)
    raw_phase = np.asarray(phase)
    if (
        not np.issubdtype(raw_amplitude.dtype, np.number)
        or not np.issubdtype(raw_phase.dtype, np.number)
        or np.iscomplexobj(raw_amplitude)
        or np.iscomplexobj(raw_phase)
    ):
        raise ValueError("amplitude and phase must be real numeric values.")
    amplitude_array, phase_array = np.broadcast_arrays(
        raw_amplitude.astype(float),
        raw_phase.astype(float),
    )
    if not np.all(np.isfinite(amplitude_array)) or not np.all(np.isfinite(phase_array)):
        raise ValueError("amplitude and phase must contain only finite values.")
    if np.any(amplitude_array < 0.0):
        raise ValueError("amplitude must be non-negative.")
    return amplitude_array * np.exp(1j * phase_array)


def make_initial_object_field(
    config: SimulationConfig,
    sigma: float = 0.35e-3,
    phase_amplitude: float = 0.5,
    phase_frequency: float = 2_000.0,
) -> np.ndarray:
    """Create the Gaussian-amplitude, sinusoidal-phase Phase 0 object field."""
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    if not math.isfinite(phase_amplitude):
        raise ValueError("phase_amplitude must be finite.")
    if not math.isfinite(phase_frequency) or phase_frequency < 0.0:
        raise ValueError("phase_frequency must be finite and non-negative.")

    x, y = config.coordinate_grid()
    amplitude = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    phase = phase_amplitude * np.cos(2.0 * np.pi * phase_frequency * x)
    return make_complex_field(amplitude, phase)


def make_reference_field(config: SimulationConfig) -> np.ndarray:
    """Create the configured tilted plane-wave reference field."""
    x, y = config.coordinate_grid()
    phase = 2.0 * np.pi * (
        config.reference_frequency_x * x + config.reference_frequency_y * y
    )
    return config.reference_amplitude * np.exp(1j * phase)


def make_hologram(object_field: np.ndarray, reference_field: np.ndarray) -> np.ndarray:
    """Record the real-valued interference intensity of two complex fields."""
    object_array = _validated_complex_field(object_field, "object_field")
    reference_array = _validated_complex_field(reference_field, "reference_field")
    if object_array.shape != reference_array.shape:
        raise ValueError("object_field and reference_field must have the same shape.")
    with np.errstate(over="ignore", invalid="ignore"):
        intensity = np.asarray(np.abs(object_array + reference_array) ** 2, dtype=float)
    if not np.all(np.isfinite(intensity)):
        raise ValueError("hologram intensity must remain finite.")
    return intensity


def reconstruct_object_field(
    hologram: np.ndarray,
    config: SimulationConfig,
    bandwidth: float = 8_000.0,
) -> np.ndarray:
    """Select the object-bearing diffraction order and reconstruct the object plane.

    ``bandwidth`` is the half-width of a rectangular passband in cycles/m.
    The selected order is the ``object * conjugate(reference)`` term, centered
    at ``(-fcx, +fcy)`` on the sampled FFT axes. The y sign accounts for rows
    running opposite to the positive physical y direction.
    """
    hologram_array = np.asarray(hologram)
    expected_shape = (config.ny, config.nx)
    if (
        hologram_array.ndim != 2
        or hologram_array.shape != expected_shape
        or not np.issubdtype(hologram_array.dtype, np.number)
        or np.iscomplexobj(hologram_array)
        or not np.all(np.isfinite(hologram_array))
    ):
        raise ValueError(f"hologram must be a finite real array with shape {expected_shape}.")
    if np.any(hologram_array < 0.0):
        raise ValueError("hologram intensity must be non-negative.")
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive.")
    if config.reference_amplitude <= 0.0:
        raise ValueError("reference_amplitude must be positive for reconstruction.")

    fcx = config.reference_frequency_x
    fcy = config.reference_frequency_y
    if abs(fcx) <= 2.0 * bandwidth and abs(fcy) <= 2.0 * bandwidth:
        raise ValueError("reference carrier orders overlap the DC band at this bandwidth.")

    fx, fy = config.frequency_grid()
    order_center_x = -fcx
    order_center_y = fcy
    if (
        order_center_x - bandwidth < fx.min()
        or order_center_x + bandwidth > fx.max()
        or order_center_y - bandwidth < fy.min()
        or order_center_y + bandwidth > fy.max()
    ):
        raise ValueError("selected diffraction order extends beyond the sampled frequency range.")

    passband = (
        (np.abs(fx - order_center_x) <= bandwidth)
        & (np.abs(fy - order_center_y) <= bandwidth)
    )
    if not np.any(passband):
        raise ValueError("bandwidth does not include any sampled frequency bins.")
    selected_order = ifft2_centered(fft2_centered(hologram_array) * passband)
    sensor_object_estimate = selected_order * make_reference_field(config)
    if not np.all(np.isfinite(sensor_object_estimate)):
        raise ValueError("reconstructed sensor field must remain finite.")
    object_estimate = propagate_angular_spectrum(
        sensor_object_estimate,
        config,
        distance=-config.propagation_distance,
    )
    if not np.all(np.isfinite(object_estimate)):
        raise ValueError("reconstructed object field must remain finite.")
    return object_estimate


def _validated_complex_field(field: np.ndarray, name: str) -> np.ndarray:
    field_array = np.asarray(field)
    if (
        field_array.ndim != 2
        or not np.issubdtype(field_array.dtype, np.number)
        or not np.all(np.isfinite(field_array))
    ):
        raise ValueError(f"{name} must be a finite 2D numeric array.")
    return field_array


def propagate_angular_spectrum(
    field: np.ndarray,
    config: SimulationConfig,
    distance: float | None = None,
    band_limited: bool = False,
) -> np.ndarray:
    """Propagate a sampled complex field with the angular spectrum method.

    Negative distances perform reverse propagation. With ``band_limited=True``,
    evanescent waves and frequencies beyond the sampled angular-spectrum limits
    are removed.
    """
    field_array = np.asarray(field)
    expected_shape = (config.ny, config.nx)
    if field_array.ndim != 2 or field_array.shape != expected_shape:
        raise ValueError(f"field must have shape {expected_shape}.")
    if not np.issubdtype(field_array.dtype, np.number) or not np.all(np.isfinite(field_array)):
        raise ValueError("field must contain only finite numeric values.")

    z = config.propagation_distance if distance is None else distance
    if not math.isfinite(z):
        raise ValueError("distance must be finite.")
    if z == 0.0:
        return np.array(field_array, dtype=complex, copy=True)

    fx, fy = config.frequency_grid()
    kx, ky = config.wave_number_grid()
    wave_number = 2.0 * np.pi / config.wavelength
    kz = np.sqrt((wave_number**2 - kx**2 - ky**2).astype(complex))

    if band_limited:
        abs_z = abs(z)
        fx_limit = 1.0 / (
            config.wavelength
            * math.sqrt(1.0 + (2.0 * abs_z / (config.nx * config.dx)) ** 2)
        )
        fy_limit = 1.0 / (
            config.wavelength
            * math.sqrt(1.0 + (2.0 * abs_z / (config.ny * config.dy)) ** 2)
        )
        propagating = fx**2 + fy**2 <= (1.0 / config.wavelength) ** 2
        sampled = (np.abs(fx) <= fx_limit) & (np.abs(fy) <= fy_limit)
        valid = propagating & sampled
        transfer = np.zeros(kz.shape, dtype=complex)
        transfer[valid] = np.exp(1j * kz[valid] * z)
    else:
        transfer = np.exp(1j * kz * z)

    return ifft2_centered(fft2_centered(field_array) * transfer)
