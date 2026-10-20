import matplotlib
import numpy as np

matplotlib.use("Agg")

from holography import SimulationConfig, make_initial_object_field
from holography.visualization import plot_propagation


def test_plot_propagation_shows_intensity_and_phase_at_each_distance():
    config = SimulationConfig(nx=16, ny=12, propagation_distance=0.01)
    field = make_initial_object_field(config)

    figure = plot_propagation(field, config, distances=np.array([0.0, 0.01]))

    assert len(figure.axes) == 4
    assert figure.axes[0].get_title() == "z = 0.0 cm"
    assert figure.axes[1].get_title() == "z = 1.0 cm"
    assert figure.axes[0].images[0].get_cmap().name == "inferno"
    assert figure.axes[2].images[0].get_cmap().name == "twilight"
    matplotlib.pyplot.close(figure)
