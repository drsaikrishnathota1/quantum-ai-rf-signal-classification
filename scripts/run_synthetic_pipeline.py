from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic RF paper pipeline end to end.")
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_iq_500.npz"))
    parser.add_argument("--classical-out", type=Path, default=Path("results/synthetic_500"))
    parser.add_argument("--cnn-out", type=Path, default=Path("results/cnn_500"))
    parser.add_argument("--quantum-out", type=Path, default=Path("results/quantum_kernel_500"))
    parser.add_argument("--cnn-epochs", type=int, default=18)
    parser.add_argument("--cnn-batch-size", type=int, default=128)
    parser.add_argument("--quantum-qubits", type=int, default=5)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-quantum", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not args.skip_data:
        run(
            [
                py,
                "scripts/generate_synthetic_iq_dataset.py",
                "--samples-per-class",
                str(args.samples_per_class),
                "--out",
                str(args.data),
            ]
        )

    run(
        [
            py,
            "scripts/train_pilot_classifiers.py",
            "--data",
            str(args.data),
            "--out",
            str(args.classical_out),
        ]
    )

    if not args.skip_cnn:
        run(
            [
                py,
                "scripts/train_cnn_iq_baseline.py",
                "--data",
                str(args.data),
                "--out",
                str(args.cnn_out),
                "--epochs",
                str(args.cnn_epochs),
                "--batch-size",
                str(args.cnn_batch_size),
            ]
        )

    if not args.skip_quantum:
        run(
            [
                py,
                "scripts/train_quantum_inspired_kernel.py",
                "--data",
                str(args.data),
                "--out",
                str(args.quantum_out),
                "--qubits",
                str(args.quantum_qubits),
            ]
        )

    run([py, "scripts/aggregate_synthetic_analysis.py"])


if __name__ == "__main__":
    main()
