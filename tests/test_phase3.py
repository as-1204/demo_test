import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from holography import (
    SimulationConfig,
    fft2_centered,
    make_hologram,
    make_initial_object_field,
    make_reference_field,
    propagate_angular_spectrum,
)
from holography.visualization import plot_hologram_recording


def test_reference_field_matches_configured_tilted_plane_wave():
    config = SimulationConfig(
        nx=32,
        ny=24,
        reference_amplitude=1.5,
        reference_frequency_x=12_500.0,
        reference_frequency_y=-6_250.0,
    )
    x, y = config.coordinate_grid()

    reference = make_reference_field(config)

    np.testing.assert_allclose(
        reference,
        1.5 * np.exp(2j * np.pi * (12_500.0 * x - 6_250.0 * y)),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_amplitude": -1.0},
        {"reference_amplitude": np.inf},
        {"reference_frequency_x": np.nan},
        {"reference_frequency_y": np.inf},
    ],
)
def test_config_rejects_invalid_reference_parameters(kwargs):
    with pytest.raises(ValueError, match="reference"):
        SimulationConfig(**kwargs)


def test_make_hologram_records_real_finite_interference_intensity():
    config = SimulationConfig(nx=64, ny=32)
    object_field = np.ones((config.ny, config.nx), dtype=complex)
    reference_field = make_reference_field(config)

    hologram = make_hologram(object_field, reference_field)
    x, _ = config.coordinate_grid()
    expected = 2.0 + 2.0 * np.cos(2.0 * np.pi * config.reference_frequency_x * x)

    assert hologram.shape == (config.ny, config.nx)
    assert np.issubdtype(hologram.dtype, np.floating)
    assert np.all(np.isfinite(hologram))
    assert np.all(hologram >= 0.0)
    np.testing.assert_allclose(hologram, expected, atol=1e-12)


@pytest.mark.parametrize(
    ("object_field", "reference_field", "message"),
    [
        (np.ones((2, 2)), np.ones((3, 2)), "same shape"),
        (np.ones((2, 2), dtype=complex) * np.nan, np.ones((2, 2)), "object_field"),
        (np.ones(2), np.ones(2), "2D"),
        (np.full((2, 2), 1e308), np.ones((2, 2)), "intensity"),
    ],
)
def test_make_hologram_rejects_invalid_fields(object_field, reference_field, message):
    with pytest.raises(ValueError, match=message):
        make_hologram(object_field, reference_field)


def test_hologram_spectrum_contains_separated_positive_and_negative_orders():
    config = SimulationConfig()
    object_field = propagate_angular_spectrum(make_initial_object_field(config), config)
    reference_field = make_reference_field(config)
    hologram = make_hologram(object_field, reference_field)
    spectrum = np.abs(fft2_centered(hologram))
    fx, fy = config.frequency_grid()
    object_bandwidth = 8_000.0
    carrier = config.reference_frequency_x
    nyquist = 1.0 / (2.0 * config.dx)

    assert object_bandwidth < carrier - object_bandwidth
    assert carrier + object_bandwidth < nyquist
    for sign in (-1.0, 1.0):
        order = (
            (sign * fx >= carrier - object_bandwidth)
            & (sign * fx <= carrier + object_bandwidth)
            & (np.abs(fy) <= object_bandwidth)
        )
        order_indices = np.argwhere(order)
        peak_index = order_indices[np.argmax(spectrum[order])]
        peak_frequency = fx[tuple(peak_index)]
        assert abs(peak_frequency - sign * carrier) <= 2.0 / (config.nx * config.dx)


def test_hologram_recording_plot_shows_fields_intensity_and_spectrum():
    config = SimulationConfig(nx=32, ny=24)
    object_field = make_initial_object_field(config)
    reference_field = make_reference_field(config)

    figure = plot_hologram_recording(object_field, reference_field, config)

    assert len(figure.axes) == 8
    assert figure.axes[0].get_title() == "Object amplitude"
    assert figure.axes[1].get_title() == "Reference amplitude"
    assert figure.axes[2].get_title() == "Sensor field amplitude"
    assert figure.axes[3].get_title() == "Normalized hologram intensity"
    assert figure.axes[7].get_title() == "log(1 + |FFT(hologram)|)"
    matplotlib.pyplot.close(figure)
