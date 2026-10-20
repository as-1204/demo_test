"""Phase 1 numerical foundation for scalar holography simulations."""

from .config import (
    SimulationConfig,
    fft2_centered,
    ifft2_centered,
    make_coordinate_grid,
    make_frequency_grid,
    make_wave_number_grid,
)

__all__ = [
    "SimulationConfig",
    "make_coordinate_grid",
    "make_frequency_grid",
    "make_wave_number_grid",
    "fft2_centered",
    "ifft2_centered",
]

__version__ = "0.1.0"
