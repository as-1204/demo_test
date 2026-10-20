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

## Record an off-axis hologram

Show the propagated object field, tilted reference field, sensor field, recorded
intensity, and centered Fourier spectrum:

```bash
python -m holography --record-hologram
```

Save the same recording visualization without opening a window:

```bash
python -m holography --save-hologram hologram.png
```

The defaults use a reference amplitude of `1` and carrier frequencies
`(20,000, 0) cycles/m`. With the default `8 um` pixel pitch, the Nyquist
frequency is `62,500 cycles/m`; assuming the object's useful spectrum is within
`8,000 cycles/m`, the two first-order bands around `+/-20,000 cycles/m` do not
overlap the DC band or each other and remain below Nyquist. This bandwidth is a
practical scene assumption, not a strict limit, so inspect the spectrum when
using sharper or higher-frequency objects.

## Reconstruct an object from a hologram

Reconstruct and display the object's normalized amplitude and wrapped phase:

```bash
python -m holography --reconstruct-hologram
```

Save the reconstruction without opening a window:

```bash
python -m holography --save-reconstruction reconstruction.png
```

The reconstruction selects the `object * conjugate(reference)` order with a
rectangular passband. Its default half-width is `8,000 cycles/m` on each axis;
change it with `--reconstruction-bandwidth-cycles-m`. The selected order must
fit inside the sampled frequency range and remain separated from DC and the
conjugate order. The selected spectrum is inverse transformed, multiplied by
the configured reference wave to remove its carrier, and reverse propagated
to the object plane. The plot normalizes amplitude to its peak and displays
phase wrapped to `[-pi, pi]` radians.

## Core API

```python
from holography import (
    SimulationConfig,
    make_complex_field,
    make_coordinate_grid,
    make_frequency_grid,
    make_hologram,
    make_initial_object_field,
    make_reference_field,
    propagate_angular_spectrum,
    reconstruct_object_field,
)

cfg = SimulationConfig(wavelength=532e-9, pixel_pitch=8e-6, nx=512, ny=512)
x, y = cfg.coordinate_grid()
fx, fy = cfg.frequency_grid()
object_field = make_initial_object_field(cfg)
sensor_field = propagate_angular_spectrum(object_field, cfg)
reference_field = make_reference_field(cfg)
hologram = make_hologram(sensor_field, reference_field)
recovered_field = propagate_angular_spectrum(sensor_field, cfg, distance=-cfg.propagation_distance)
reconstructed_field = reconstruct_object_field(hologram, cfg)
```

The coordinate convention follows the Phase 0 roadmap:

- `x[j] = (j - floor(Nx / 2)) * dx`
- `y[i] = (floor(Ny / 2) - i) * dy`
- `fftshift(fftfreq(N, d))` defines the frequency grid in cycles/m
- `kx = 2 * pi * fx` and `ky = 2 * pi * fy`
- `make_complex_field(amplitude, phase)` uses phase in radians and returns `amplitude * exp(1j * phase)`
- `make_initial_object_field(cfg)` creates the documented Gaussian-amplitude, sinusoidal-phase sample
- `propagate_angular_spectrum(field, cfg, distance=None)` uses the configured distance by default; a negative distance propagates in reverse
- `make_reference_field(cfg)` creates `reference_amplitude * exp(1j * 2*pi*(fcx*x + fcy*y))`, with carrier frequencies in cycles/m
- `make_hologram(object_field, reference_field)` returns the finite real intensity `abs(object_field + reference_field)**2`
- `reconstruct_object_field(hologram, cfg, bandwidth=8_000)` selects and recenters the object-bearing order, then reverse propagates it to the object plane
- Set `band_limited=True` to remove evanescent waves and frequencies outside the sampled angular-spectrum limits

## Notes

Propagation uses the centered FFT convention from the roadmap. Reverse propagation is numerically stable for the default sampled grid, which contains only propagating frequencies; evanescent components are suppressed when band limiting is enabled.
