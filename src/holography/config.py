from __future__ import annotations

from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if self.wavelength <= 0.0:
            raise ValueError("wavelength must be positive.")
        if self.pixel_pitch <= 0.0:
            raise ValueError("pixel_pitch must be positive.")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive integers.")

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
