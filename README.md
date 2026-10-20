# demo_test

This repository contains the Phase 1 numerical foundation for a scalar holography simulation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[dev]
```

## Run the tests

```bash
pytest
```

## Core API

```python
from holography import SimulationConfig, make_coordinate_grid, make_frequency_grid

cfg = SimulationConfig(wavelength=532e-9, pixel_pitch=8e-6, nx=512, ny=512)
x, y = cfg.coordinate_grid()
fx, fy = cfg.frequency_grid()
```

The coordinate convention follows the Phase 0 roadmap:

- `x[j] = (j - floor(Nx / 2)) * dx`
- `y[i] = (floor(Ny / 2) - i) * dy`
- `fftshift(fftfreq(N, d))` defines the frequency grid in cycles/m
- `kx = 2 * pi * fx` and `ky = 2 * pi * fy`

## Notes

The implementation intentionally keeps only the minimal numerical foundation required for later propagation and hologram phases.
