from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from generate_synthetic_iq_dataset import stress_signal


STRESS_CONDITIONS = [
    "clean",
    "low_snr",
    "narrowband_jam",
    "broadband_jam",
    "frequency_offset",
    "multipath",
    "impulsive_noise",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add project stress-condition arrays to a clean IQ NPZ dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def stack_complex(z: np.ndarray) -> np.ndarray:
    return np.stack([z.real, z.imag], axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data = np.load(args.input, allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)

    payload = {
        "X": x,
        "y": y,
        "snr_db": np.asarray(data["snr_db"], dtype=np.float32),
        "modulations": np.asarray(data["modulations"]),
        "stress_conditions": np.asarray(STRESS_CONDITIONS),
    }

    complex_rows = x[:, 0, :] + 1j * x[:, 1, :]
    for condition in STRESS_CONDITIONS:
        stress_x = np.empty_like(x, dtype=np.float32)
        for idx, sample in enumerate(complex_rows):
            stress_x[idx] = stack_complex(stress_signal(sample, condition, rng))
        payload[f"X_{condition}"] = stress_x
        payload[f"y_{condition}"] = y.copy()
        print(f"Added condition: {condition}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"Wrote {args.out}")
    print(f"X shape: {x.shape}")
    print(f"Stress conditions: {STRESS_CONDITIONS}")


if __name__ == "__main__":
    main()
