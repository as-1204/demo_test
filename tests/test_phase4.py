import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from holography import (
    SimulationConfig,
    make_hologram,
    make_initial_object_field,
    make_reference_field,
    propagate_angular_spectrum,
    reconstruct_object_field,
)
from holography.visualization import plot_reconstruction


@pytest.mark.parametrize(
    "carrier",
    [
        {"reference_frequency_x": 20_000.0, "reference_frequency_y": 0.0},
        {"reference_frequency_x": 0.0, "reference_frequency_y": 20_000.0},
    ],
)
def test_reconstruction_recovers_object_amplitude_and_phase(carrier):
    config = SimulationConfig(nx=256, ny=256, **carrier)
    object_field = make_initial_object_field(config)
    sensor_field = propagate_angular_spectrum(object_field, config)
    hologram = make_hologram(sensor_field, make_reference_field(config))

    reconstructed = reconstruct_object_field(hologram, config)

    assert reconstructed.shape == object_field.shape
    assert np.all(np.isfinite(reconstructed))
    np.testing.assert_allclose(
        np.abs(reconstructed),
        np.abs(object_field),
        atol=0.003,
    )
    relevant_phase = np.abs(object_field) > 0.1
    phase_difference = np.angle(reconstructed[relevant_phase] * np.conj(object_field[relevant_phase]))
    assert np.max(np.abs(phase_difference)) < 0.01


@pytest.mark.parametrize(
    ("hologram", "config", "bandwidth", "message"),
    [
        (np.ones((2, 4)), SimulationConfig(nx=4, ny=3), 8_000.0, "shape"),
        (np.full((3, 4), np.nan), SimulationConfig(nx=4, ny=3), 8_000.0, "finite real"),
        (np.full((3, 4), 1j), SimulationConfig(nx=4, ny=3), 8_000.0, "finite real"),
        (np.full((3, 4), -1.0), SimulationConfig(nx=4, ny=3), 8_000.0, "non-negative"),
        (
            np.ones((3, 4)),
            SimulationConfig(nx=4, ny=3),
            0.0,
            "bandwidth",
        ),
        (
            np.ones((3, 4)),
            SimulationConfig(nx=4, ny=3, reference_frequency_x=10_000.0),
            8_000.0,
            "overlap",
        ),
        (
            np.ones((3, 4)),
            SimulationConfig(nx=4, ny=3, reference_frequency_x=60_000.0),
            8_000.0,
            "frequency range",
        ),
        (
            np.ones((3, 4)),
            SimulationConfig(nx=4, ny=3, reference_amplitude=0.0),
            8_000.0,
            "reference_amplitude",
        ),
        (
            np.ones((3, 4)),
            SimulationConfig(nx=4, ny=3),
            1.0,
            "sampled frequency bins",
        ),
    ],
)
def test_reconstruction_rejects_invalid_inputs(hologram, config, bandwidth, message):
    with pytest.raises(ValueError, match=message):
        reconstruct_object_field(hologram, config, bandwidth=bandwidth)


def test_reconstruction_plot_labels_normalization_and_wrapped_phase():
    config = SimulationConfig(nx=32, ny=24)
    field = make_initial_object_field(config)

    figure = plot_reconstruction(field, config)

    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Reconstructed amplitude (normalized to peak)"
    assert figure.axes[1].get_title() == "Reconstructed wrapped phase [-pi, pi] (rad)"
    assert figure.axes[0].images[0].get_clim() == (0.0, 1.0)
    assert figure.axes[1].images[0].get_clim() == (-np.pi, np.pi)
    assert figure.axes[0].images[0].origin == "upper"
    matplotlib.pyplot.close(figure)
