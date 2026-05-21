from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pilot RF classifiers on synthetic IQ data.")
    parser.add_argument("--data", type=Path, default=Path("data/pilot_iq.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/pilot"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def iq_to_complex(x: np.ndarray) -> np.ndarray:
    return x[:, 0, :] + 1j * x[:, 1, :]


def extract_features(x: np.ndarray) -> np.ndarray:
    z = iq_to_complex(x)
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z), axis=1)
    power = np.mean(amp**2, axis=1)
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft(z, axis=1), axes=1))
    bins = np.linspace(-0.5, 0.5, z.shape[1], endpoint=False)
    spectral_power = fft_mag + 1e-8
    spectral_centroid = np.sum(spectral_power * bins, axis=1) / np.sum(spectral_power, axis=1)
    spectral_bandwidth = np.sqrt(
        np.sum(spectral_power * (bins - spectral_centroid[:, None]) ** 2, axis=1)
        / np.sum(spectral_power, axis=1)
    )
    features = np.column_stack(
        [
            np.mean(z.real, axis=1),
            np.std(z.real, axis=1),
            np.mean(z.imag, axis=1),
            np.std(z.imag, axis=1),
            np.mean(amp, axis=1),
            np.std(amp, axis=1),
            np.percentile(amp, 25, axis=1),
            np.percentile(amp, 75, axis=1),
            np.mean(phase, axis=1),
            np.std(phase, axis=1),
            power,
            np.mean(amp**4, axis=1),
            spectral_centroid,
            spectral_bandwidth,
            np.max(fft_mag, axis=1) / np.mean(fft_mag, axis=1),
        ]
    )
    return np.nan_to_num(features).astype(np.float32)


def evaluate(model_name: str, model, x: np.ndarray, y: np.ndarray, condition: str) -> dict[str, float | str]:
    pred = model.predict(extract_features(x))
    return {
        "model": model_name,
        "condition": condition,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data, allow_pickle=True)
    x = data["X"]
    y = data["y"]
    modulations = [str(v) for v in data["modulations"]]
    stress_conditions = [str(v) for v in data["stress_conditions"]]

    x_feat = extract_features(x)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=args.seed,
        stratify=y,
    )
    x_train, x_test = x_feat[train_idx], x_feat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1500, n_jobs=1, C=2.0)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=args.seed,
            n_jobs=1,
            min_samples_leaf=2,
        ),
        "rbf_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(C=8.0, gamma="scale")),
            ]
        ),
    }

    records = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        clean_pred = model.predict(x_test)
        records.append(
            {
                "model": model_name,
                "condition": "heldout_clean",
                "accuracy": float(accuracy_score(y_test, clean_pred)),
                "macro_f1": float(f1_score(y_test, clean_pred, average="macro")),
            }
        )
        joblib.dump(model, args.out / f"{model_name}.joblib")

        for condition in stress_conditions:
            records.append(
                evaluate(
                    model_name,
                    model,
                    data[f"X_{condition}"][test_idx],
                    data[f"y_{condition}"][test_idx],
                    condition,
                )
            )

        cm = confusion_matrix(y_test, clean_pred)
        np.savetxt(args.out / f"confusion_matrix_{model_name}.csv", cm, delimiter=",", fmt="%d")

    df = pd.DataFrame(records)
    clean_refs = df[df["condition"] == "clean"].set_index("model")
    drop_rows = []
    for _, row in df.iterrows():
        if row["condition"] in {"heldout_clean", "clean"}:
            continue
        ref = clean_refs.loc[row["model"]]
        drop_rows.append(
            {
                "model": row["model"],
                "condition": row["condition"],
                "accuracy": row["accuracy"],
                "accuracy_drop_pct": float((ref["accuracy"] - row["accuracy"]) / ref["accuracy"] * 100),
                "macro_f1": row["macro_f1"],
                "macro_f1_drop_pct": float((ref["macro_f1"] - row["macro_f1"]) / ref["macro_f1"] * 100),
            }
        )

    df.to_csv(args.out / "pilot_metrics.csv", index=False)
    pd.DataFrame(drop_rows).to_csv(args.out / "pilot_robustness_drop.csv", index=False)

    summary = {
        "dataset": str(args.data),
        "num_examples": int(len(y)),
        "num_classes": int(len(modulations)),
        "modulations": modulations,
        "stress_conditions": stress_conditions,
        "best_clean_accuracy": float(df[df["condition"] == "clean"]["accuracy"].max()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.round(4).to_string(index=False))
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
