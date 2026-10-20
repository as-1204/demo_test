from __future__ import annotations

import argparse

from .config import SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the Phase 1 holography grid configuration.")
    parser.add_argument("--wavelength-nm", type=float, default=532.0, help="Wavelength in nanometers.")
    parser.add_argument("--pixel-pitch-um", type=float, default=8.0, help="Sensor pixel pitch in micrometers.")
    parser.add_argument("--nx", type=int, default=512, help="Number of samples along x.")
    parser.add_argument("--ny", type=int, default=512, help="Number of samples along y.")
    parser.add_argument("--distance-m", type=float, default=0.10, help="Propagation distance in meters.")
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


if __name__ == "__main__":
    main()
