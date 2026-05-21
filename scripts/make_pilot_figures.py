from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = ["#2E74B5", "#70AD47", "#C55A11", "#8064A2", "#5B9BD5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create simple pilot figures without Matplotlib.")
    parser.add_argument("--metrics", type=Path, default=Path("results/pilot/pilot_metrics.csv"))
    parser.add_argument("--drops", type=Path, default=Path("results/pilot/pilot_robustness_drop.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/pilot/figures"))
    return parser.parse_args()


def load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def bar_chart(labels: list[str], values: list[float], title: str, ylabel: str, out: Path) -> None:
    width, height = 1400, 820
    margin_l, margin_r, margin_t, margin_b = 150, 80, 110, 180
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(38, True)
    label_font = load_font(22)
    small_font = load_font(19)
    draw.text((width // 2, 45), title, fill="#111111", font=title_font, anchor="mm")
    draw.text((35, height // 2), ylabel, fill="#111111", font=label_font, anchor="mm")
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x0, y0 = margin_l, height - margin_b
    draw.line((x0, margin_t, x0, y0), fill="#333333", width=3)
    draw.line((x0, y0, width - margin_r, y0), fill="#333333", width=3)
    max_value = max(max(values), 1e-9)
    top = max(1.0, max_value * 1.15)
    for i in range(6):
        val = top * i / 5
        y = y0 - (val / top) * plot_h
        draw.line((x0 - 8, y, width - margin_r, y), fill="#E6E6E6", width=1)
        draw.text((x0 - 15, y), f"{val:.2f}", fill="#333333", font=small_font, anchor="rm")
    bar_gap = 24
    bar_w = max(30, (plot_w - bar_gap * (len(values) + 1)) / len(values))
    for idx, (label, value) in enumerate(zip(labels, values)):
        x1 = x0 + bar_gap + idx * (bar_w + bar_gap)
        x2 = x1 + bar_w
        y1 = y0 - (value / top) * plot_h
        draw.rectangle((x1, y1, x2, y0), fill=COLORS[idx % len(COLORS)])
        draw.text(((x1 + x2) / 2, y1 - 18), f"{value:.2f}", fill="#111111", font=small_font, anchor="mm")
        label_lines = label.replace("_", "\n").split("\n")
        for line_idx, line in enumerate(label_lines[:4]):
            draw.text(((x1 + x2) / 2, y0 + 26 + line_idx * 24), line, fill="#111111", font=small_font, anchor="mt")
    img.save(out, dpi=(300, 300))


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics)
    drops = pd.read_csv(args.drops)

    clean = metrics[metrics["condition"] == "clean"].sort_values("accuracy", ascending=False)
    bar_chart(
        clean["model"].tolist(),
        clean["accuracy"].tolist(),
        "Pilot Clean RF Classification Accuracy",
        "Accuracy",
        args.out / "pilot_clean_accuracy.png",
    )

    worst = drops.sort_values("accuracy_drop_pct", ascending=False).head(8)
    labels = [f"{r.model}\n{r.condition}" for r in worst.itertuples()]
    bar_chart(
        labels,
        worst["accuracy_drop_pct"].tolist(),
        "Pilot Robustness Drop Under RF Stress",
        "Accuracy drop (%)",
        args.out / "pilot_robustness_drop.png",
    )
    print(f"Saved figures to {args.out}")


if __name__ == "__main__":
    main()
