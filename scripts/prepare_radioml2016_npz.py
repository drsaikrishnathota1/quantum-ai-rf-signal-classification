from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RadioML2016.10A pickle data into project NPZ format.")
    parser.add_argument("--input", type=Path, required=True, help="Path to RML2016.10a_dict.pkl.")
    parser.add_argument("--out", type=Path, default=Path("data/radioml/radioml2016_10a_clean.npz"))
    parser.add_argument("--max-examples-per-mod-snr", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except TypeError:
            f.seek(0)
            return pickle.load(f)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    raw = load_pickle(args.input)
    keys = sorted(raw.keys(), key=lambda item: (str(item[0]), int(item[1])))
    modulations = sorted({str(mod) for mod, _snr in keys})
    mod_to_id = {mod: idx for idx, mod in enumerate(modulations)}

    x_rows = []
    y_rows = []
    snr_rows = []
    for mod, snr in keys:
        arr = np.asarray(raw[(mod, snr)], dtype=np.float32)
        if args.max_examples_per_mod_snr and len(arr) > args.max_examples_per_mod_snr:
            keep = rng.choice(len(arr), size=args.max_examples_per_mod_snr, replace=False)
            arr = arr[keep]
        x_rows.append(arr)
        y_rows.append(np.full(len(arr), mod_to_id[str(mod)], dtype=np.int64))
        snr_rows.append(np.full(len(arr), float(snr), dtype=np.float32))

    x = np.concatenate(x_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)
    snr_db = np.concatenate(snr_rows, axis=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=x,
        y=y,
        snr_db=snr_db,
        modulations=np.asarray(modulations),
    )
    print(f"Wrote {args.out}")
    print(f"X shape: {x.shape}")
    print(f"Classes: {modulations}")
    print(f"SNR values: {sorted(set(float(v) for v in snr_db))}")


if __name__ == "__main__":
    main()
