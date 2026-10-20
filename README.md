# demo_test

This repository contains a scalar holography simulation with complex object-field generation and angular-spectrum propagation.

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

## Visualize propagation

Show the object field's normalized intensity and phase at five propagation
distances from the object plane to the sensor plane:

```bash
python -m holography --visualize
```

Save the same visualization to an image file without opening a window:

```bash
python -m holography --save-plot propagation.png
```

Use both options to save the image and open it interactively. Grid parameters
such as `--nx`, `--ny`, and `--distance-m` can be changed on the command line.

## Core API

```python
from holography import (
    SimulationConfig,
    make_complex_field,
    make_coordinate_grid,
    make_frequency_grid,
    make_initial_object_field,
    propagate_angular_spectrum,
)

cfg = SimulationConfig(wavelength=532e-9, pixel_pitch=8e-6, nx=512, ny=512)
x, y = cfg.coordinate_grid()
fx, fy = cfg.frequency_grid()
object_field = make_initial_object_field(cfg)
sensor_field = propagate_angular_spectrum(object_field, cfg)
recovered_field = propagate_angular_spectrum(sensor_field, cfg, distance=-cfg.propagation_distance)
```

The coordinate convention follows the Phase 0 roadmap:

- `x[j] = (j - floor(Nx / 2)) * dx`
- `y[i] = (floor(Ny / 2) - i) * dy`
- `fftshift(fftfreq(N, d))` defines the frequency grid in cycles/m
- `kx = 2 * pi * fx` and `ky = 2 * pi * fy`
- `make_complex_field(amplitude, phase)` uses phase in radians and returns `amplitude * exp(1j * phase)`
- `make_initial_object_field(cfg)` creates the documented Gaussian-amplitude, sinusoidal-phase sample
- `propagate_angular_spectrum(field, cfg, distance=None)` uses the configured distance by default; a negative distance propagates in reverse
- Set `band_limited=True` to remove evanescent waves and frequencies outside the sampled angular-spectrum limits

## Notes

Propagation uses the centered FFT convention from the roadmap. Reverse propagation is numerically stable for the default sampled grid, which contains only propagating frequencies; evanescent components are suppressed when band limiting is enabled.
