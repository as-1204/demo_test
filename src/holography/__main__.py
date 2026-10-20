from __future__ import annotations

import argparse

from .config import SimulationConfig, make_initial_object_field


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the holography simulation grid configuration.")
    parser.add_argument("--wavelength-nm", type=float, default=532.0, help="Wavelength in nanometers.")
    parser.add_argument("--pixel-pitch-um", type=float, default=8.0, help="Sensor pixel pitch in micrometers.")
    parser.add_argument("--nx", type=int, default=512, help="Number of samples along x.")
    parser.add_argument("--ny", type=int, default=512, help="Number of samples along y.")
    parser.add_argument("--distance-m", type=float, default=0.10, help="Propagation distance in meters.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show intensity and phase at several propagation distances.",
    )
    parser.add_argument(
        "--save-plot",
        metavar="PATH",
        help="Save the propagation visualization to an image file instead of opening a window.",
    )
    parser.add_argument(
        "--record-hologram",
        action="store_true",
        help="Show the off-axis hologram, complex sensor fields, and Fourier spectrum.",
    )
    parser.add_argument(
        "--save-hologram",
        metavar="PATH",
        help="Save the off-axis hologram recording visualization to an image file.",
    )
    parser.add_argument(
        "--reconstruct-hologram",
        action="store_true",
        help="Reconstruct the object-plane amplitude and wrapped phase from a recorded hologram.",
    )
    parser.add_argument(
        "--save-reconstruction",
        metavar="PATH",
        help="Save the digital reconstruction visualization to an image file.",
    )
    parser.add_argument(
        "--reconstruction-bandwidth-cycles-m",
        type=float,
        default=8_000.0,
        help="Half-width of the selected diffraction-order passband in cycles/m.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = SimulationConfig(
        wavelength=args.wavelength_nm * 1e-9,
        pixel_pitch=args.pixel_pitch_um * 1e-6,
        nx=args.nx,
        ny=args.ny,
        propagation_distance=args.distance_m,
    )
    x, y = cfg.coordinate_grid()
    fx, fy = cfg.frequency_grid()
    print(f"Wavelength: {cfg.wavelength:.3e} m")
    print(f"Pixel pitch: {cfg.pixel_pitch:.3e} m")
    print(f"Grid size: {cfg.nx} x {cfg.ny}")
    print(f"x range: [{x.min():.3e}, {x.max():.3e}] m")
    print(f"y range: [{y.min():.3e}, {y.max():.3e}] m")
    print(f"fx range: [{fx.min():.3e}, {fx.max():.3e}] cycles/m")
    print(f"fy range: [{fy.min():.3e}, {fy.max():.3e}] cycles/m")

    if (
        args.visualize
        or args.save_plot
        or args.record_hologram
        or args.save_hologram
        or args.reconstruct_hologram
        or args.save_reconstruction
    ):
        from matplotlib import pyplot as plt

        object_field = make_initial_object_field(cfg)
        if args.visualize or args.save_plot:
            from .visualization import plot_propagation

            figure = plot_propagation(object_field, cfg)
            if args.save_plot:
                figure.savefig(args.save_plot, dpi=150)
                print(f"Saved propagation plot: {args.save_plot}")
            if args.visualize:
                plt.show()
            else:
                plt.close(figure)

        if (
            args.record_hologram
            or args.save_hologram
            or args.reconstruct_hologram
            or args.save_reconstruction
        ):
            from .config import make_hologram, make_reference_field, propagate_angular_spectrum
            from .visualization import plot_hologram_recording

            sensor_field = propagate_angular_spectrum(object_field, cfg)
            reference_field = make_reference_field(cfg)
            if args.record_hologram or args.save_hologram:
                figure = plot_hologram_recording(sensor_field, reference_field, cfg)
                if args.save_hologram:
                    figure.savefig(args.save_hologram, dpi=150)
                    print(f"Saved hologram visualization: {args.save_hologram}")
                if args.record_hologram:
                    plt.show()
                else:
                    plt.close(figure)

            if args.reconstruct_hologram or args.save_reconstruction:
                from .config import reconstruct_object_field
                from .visualization import plot_reconstruction

                hologram = make_hologram(sensor_field, reference_field)
                reconstructed_field = reconstruct_object_field(
                    hologram,
                    cfg,
                    bandwidth=args.reconstruction_bandwidth_cycles_m,
                )
                figure = plot_reconstruction(reconstructed_field, cfg)
                if args.save_reconstruction:
                    figure.savefig(args.save_reconstruction, dpi=150)
                    print(f"Saved reconstruction visualization: {args.save_reconstruction}")
                if args.reconstruct_hologram:
                    plt.show()
                else:
                    plt.close(figure)


if __name__ == "__main__":
    main()
