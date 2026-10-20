import numpy as np

from holography import (
    SimulationConfig,
    fft2_centered,
    ifft2_centered,
    make_coordinate_grid,
    make_frequency_grid,
    make_wave_number_grid,
)


def test_default_config_respects_phase_zero_values():
    cfg = SimulationConfig()
    assert cfg.wavelength == 532e-9
    assert cfg.pixel_pitch == 8e-6
    assert cfg.nx == 512
    assert cfg.ny == 512
    assert cfg.propagation_distance == 0.10
    assert cfg.dx == cfg.pixel_pitch
    assert cfg.dy == cfg.pixel_pitch


def test_coordinate_grid_shape_and_spacing():
    x, y = make_coordinate_grid(8, 6, dx=2.0, dy=3.0)

    assert x.shape == (6, 8)
    assert y.shape == (6, 8)
    assert x[0, 4] == 0.0
    assert y[3, 0] == 0.0
    np.testing.assert_allclose(np.diff(x[0]), np.full(7, 2.0))
    np.testing.assert_allclose(np.diff(y[:, 0]), np.full(5, -3.0))


def test_frequency_grid_is_zero_centered_and_in_cycles_per_meter():
    fx, fy = make_frequency_grid(8, 6, dx=0.5, dy=1.0 / 3.0)

    assert fx.shape == (6, 8)
    assert fy.shape == (6, 8)
    assert fx[0, 4] == 0.0
    assert fy[3, 0] == 0.0
    assert np.all(fx[0, :4] < 0.0)
    assert np.all(fx[0, 5:] > 0.0)
    assert np.all(fy[:3, 0] < 0.0)
    assert np.all(fy[4:, 0] > 0.0)


def test_wave_number_grid_is_two_pi_times_frequency_grid():
    cx, cy = make_wave_number_grid(4, 5, dx=1.0, dy=2.0)
    fx, fy = make_frequency_grid(4, 5, dx=1.0, dy=2.0)

    np.testing.assert_allclose(cx, 2.0 * np.pi * fx)
    np.testing.assert_allclose(cy, 2.0 * np.pi * fy)


def test_fft_shift_round_trip_preserves_field():
    field = np.arange(16, dtype=float).reshape(4, 4) + 1j * np.arange(16, dtype=float).reshape(4, 4)
    transformed = fft2_centered(field)
    recovered = ifft2_centered(transformed)

    np.testing.assert_allclose(recovered, field)
    assert transformed.shape == field.shape
