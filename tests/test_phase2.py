import numpy as np
import pytest

from holography import (
    SimulationConfig,
    make_complex_field,
    make_initial_object_field,
    propagate_angular_spectrum,
)


def test_make_complex_field_combines_amplitude_and_phase():
    amplitude = np.array([[0.0, 1.0], [0.5, 2.0]])
    phase = np.array([[0.2, -0.7], [1.1, np.pi]])

    field = make_complex_field(amplitude, phase)

    np.testing.assert_allclose(np.abs(field), amplitude)
    np.testing.assert_allclose(field, amplitude * np.exp(1j * phase))


def test_make_complex_field_rejects_negative_amplitude():
    with pytest.raises(ValueError, match="non-negative"):
        make_complex_field(np.array([1.0, -0.1]), np.zeros(2))


def test_initial_object_field_matches_phase_zero_scene():
    config = SimulationConfig(nx=32, ny=24)
    x, y = config.coordinate_grid()

    field = make_initial_object_field(config)
    expected_amplitude = np.exp(-(x**2 + y**2) / (2.0 * (0.35e-3) ** 2))
    expected_phase = 0.5 * np.cos(2.0 * np.pi * 2_000.0 * x)

    assert field.shape == (24, 32)
    np.testing.assert_allclose(np.abs(field), expected_amplitude)
    np.testing.assert_allclose(field, expected_amplitude * np.exp(1j * expected_phase))


def test_zero_distance_propagation_preserves_field():
    config = SimulationConfig(nx=16, ny=12)
    field = np.arange(192).reshape(12, 16) * (1.0 + 0.25j)

    propagated = propagate_angular_spectrum(field, config, distance=0.0)

    np.testing.assert_array_equal(propagated, field)
    assert np.iscomplexobj(propagated)


def test_plane_wave_propagates_by_expected_global_phase():
    config = SimulationConfig(wavelength=532e-9, nx=16, ny=12)
    field = np.ones((config.ny, config.nx), dtype=complex)
    distance = 0.013

    propagated = propagate_angular_spectrum(field, config, distance=distance)

    expected = np.exp(1j * 2.0 * np.pi * distance / config.wavelength)
    np.testing.assert_allclose(propagated, expected * field, atol=1e-12)


def test_forward_then_reverse_propagation_recovers_field():
    config = SimulationConfig(nx=32, ny=24)
    rng = np.random.default_rng(42)
    field = rng.normal(size=(config.ny, config.nx)) + 1j * rng.normal(
        size=(config.ny, config.nx)
    )

    forward = propagate_angular_spectrum(field, config, distance=0.025)
    recovered = propagate_angular_spectrum(forward, config, distance=-0.025)

    np.testing.assert_allclose(recovered, field, rtol=1e-12, atol=1e-12)


def test_band_limited_propagation_removes_evanescent_wave():
    config = SimulationConfig(
        wavelength=1e-6,
        pixel_pitch=0.2e-6,
        nx=16,
        ny=16,
    )
    x, _ = config.coordinate_grid()
    evanescent_frequency = 1.25 / config.wavelength
    field = np.exp(2j * np.pi * evanescent_frequency * x)

    with np.errstate(over="raise", invalid="raise"):
        propagated = propagate_angular_spectrum(
            field,
            config,
            distance=-0.1e-6,
            band_limited=True,
        )

    np.testing.assert_allclose(propagated, 0.0, atol=1e-12)


def test_band_limited_propagation_removes_aliased_frequency():
    config = SimulationConfig(nx=32, ny=32)
    x, _ = config.coordinate_grid()
    frequency = 1.0 / (config.nx * config.dx)
    field = np.exp(2j * np.pi * frequency * x)

    propagated = propagate_angular_spectrum(
        field,
        config,
        distance=config.propagation_distance,
        band_limited=True,
    )

    np.testing.assert_allclose(propagated, 0.0, atol=1e-12)


def test_propagation_rejects_wrong_shape():
    config = SimulationConfig(nx=16, ny=12)

    with pytest.raises(ValueError, match="shape"):
        propagate_angular_spectrum(np.ones((16, 12)), config)
