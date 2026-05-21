from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class IQCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact 1D CNN baseline on raw IQ samples.")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_iq_500.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/cnn_500"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Training device. Use auto for CUDA on RunPod, MPS on Apple Silicon, then CPU fallback.",
    )
    return parser.parse_args()


def stratified_limit_indices(
    indices: np.ndarray,
    y: np.ndarray,
    max_examples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_examples <= 0 or len(indices) <= max_examples:
        return indices
    classes = np.unique(y[indices])
    per_class = max(1, max_examples // len(classes))
    selected = []
    for cls in classes:
        cls_idx = indices[y[indices] == cls]
        take = min(per_class, len(cls_idx))
        selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) < max_examples:
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        extra_count = min(max_examples - len(selected), len(remaining))
        if extra_count > 0:
            selected = np.concatenate([selected, rng.choice(remaining, size=extra_count, replace=False)])
    rng.shuffle(selected)
    return selected


def resolve_device(name: str) -> torch.device:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
    if name == "mps" and not mps_available:
        raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false.")
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = make_loader(x, np.zeros(len(x), dtype=np.int64), batch_size, False)
    for xb, _ in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def evaluate(
    model: nn.Module,
    condition: str,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | str | int]:
    pred = predict(model, x, batch_size, device)
    return {
        "model": "iq_cnn",
        "condition": condition,
        "examples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.data, allow_pickle=True)
    x = data["X"]
    y = data["y"]
    snr_db = data["snr_db"]
    modulations = [str(v) for v in data["modulations"]]
    stress_conditions = [str(v) for v in data["stress_conditions"]]

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=args.seed,
        stratify=y,
    )
    train_idx = stratified_limit_indices(train_idx, y, args.max_train_examples, rng)
    test_idx = stratified_limit_indices(test_idx, y, args.max_test_examples, rng)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    model = IQCNN(num_classes=len(modulations)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = make_loader(x_train, y_train, args.batch_size, True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
        train_loss = total_loss / len(y_train)
        clean_eval = evaluate(model, "heldout_clean", x_test, y_test, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **clean_eval})
        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"heldout_acc={clean_eval['accuracy']:.4f} heldout_f1={clean_eval['macro_f1']:.4f}"
        )

    records = [evaluate(model, "heldout_clean", x_test, y_test, args.batch_size, device)]
    for condition in stress_conditions:
        records.append(
            evaluate(
                model,
                condition,
                data[f"X_{condition}"][test_idx],
                data[f"y_{condition}"][test_idx],
                args.batch_size,
                device,
            )
        )

    metrics = pd.DataFrame(records)
    clean = metrics[metrics["condition"] == "clean"].iloc[0]
    drop_rows = []
    for _, row in metrics.iterrows():
        if row["condition"] in {"heldout_clean", "clean"}:
            continue
        drop_rows.append(
            {
                "model": row["model"],
                "condition": row["condition"],
                "accuracy": row["accuracy"],
                "accuracy_drop_pct": float((clean["accuracy"] - row["accuracy"]) / clean["accuracy"] * 100),
                "macro_f1": row["macro_f1"],
                "macro_f1_drop_pct": float((clean["macro_f1"] - row["macro_f1"]) / clean["macro_f1"] * 100),
            }
        )

    pred_clean = predict(model, x_test, args.batch_size, device)
    np.savetxt(args.out / "confusion_matrix_iq_cnn.csv", confusion_matrix(y_test, pred_clean), delimiter=",", fmt="%d")
    pd.DataFrame(history).to_csv(args.out / "cnn_training_history.csv", index=False)
    metrics.to_csv(args.out / "cnn_metrics.csv", index=False)
    pd.DataFrame(drop_rows).to_csv(args.out / "cnn_robustness_drop.csv", index=False)
    torch.save(model.state_dict(), args.out / "iq_cnn_state_dict.pt")

    snr_rows = []
    pred = pred_clean
    for snr in sorted(np.unique(snr_db[test_idx])):
        mask = snr_db[test_idx] == snr
        snr_rows.append(
            {
                "snr_db": float(snr),
                "examples": int(mask.sum()),
                "accuracy": float(accuracy_score(y_test[mask], pred[mask])),
                "macro_f1": float(f1_score(y_test[mask], pred[mask], average="macro")),
            }
        )
    pd.DataFrame(snr_rows).to_csv(args.out / "cnn_accuracy_by_snr.csv", index=False)

    summary = {
        "dataset": str(args.data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "modulations": modulations,
        "clean_accuracy": float(clean["accuracy"]),
        "clean_macro_f1": float(clean["macro_f1"]),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.round(4).to_string(index=False))
    print(f"Saved CNN outputs to {args.out}")


if __name__ == "__main__":
    main()
