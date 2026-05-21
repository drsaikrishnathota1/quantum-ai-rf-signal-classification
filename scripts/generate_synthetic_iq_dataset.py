from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MODULATIONS = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "BFSK", "AM-DSB", "FM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic RF/IQ modulation dataset.")
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--iq-length", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("data/pilot_iq.npz"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def normalize(x: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(x) ** 2)
    if power <= 1e-12:
        return x
    return x / np.sqrt(power)


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(np.abs(x) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )
    return x + noise


def psk(order: int, n: int, rng: np.random.Generator) -> np.ndarray:
    symbols = rng.integers(0, order, n)
    phase = 2 * np.pi * symbols / order
    return np.exp(1j * phase)


def qam(order: int, n: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.sqrt(order))
    vals = np.arange(-(side - 1), side, 2)
    i = rng.choice(vals, n)
    q = rng.choice(vals, n)
    return normalize(i + 1j * q)


def bfsk(n: int, rng: np.random.Generator) -> np.ndarray:
    bits = rng.integers(0, 2, n)
    freq0 = 0.06
    freq1 = 0.18
    phase = np.cumsum(2 * np.pi * np.where(bits == 0, freq0, freq1))
    return np.exp(1j * phase)


def am_dsb(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    tone_freq = rng.uniform(0.02, 0.08)
    carrier_freq = rng.uniform(0.16, 0.23)
    message = 1.0 + 0.6 * np.sin(2 * np.pi * tone_freq * t + rng.uniform(0, 2 * np.pi))
    carrier = np.exp(1j * (2 * np.pi * carrier_freq * t + rng.uniform(0, 2 * np.pi)))
    return normalize(message * carrier)


def fm(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    message_freq = rng.uniform(0.015, 0.07)
    message = np.sin(2 * np.pi * message_freq * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * 0.12 * t + 2.5 * np.cumsum(message) / n
    return normalize(np.exp(1j * phase))


def apply_channel(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    phase = rng.uniform(0, 2 * np.pi)
    y *= np.exp(1j * phase)
    offset = rng.uniform(-0.015, 0.015)
    y *= np.exp(1j * 2 * np.pi * offset * np.arange(len(y)))
    if rng.random() < 0.4:
        delay = rng.integers(1, 5)
        gain = rng.uniform(0.05, 0.25) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        z = y.copy()
        z[delay:] += gain * y[:-delay]
        y = z
    return normalize(y)


def generate_signal(modulation: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if modulation == "BPSK":
        x = psk(2, n, rng)
    elif modulation == "QPSK":
        x = psk(4, n, rng)
    elif modulation == "8PSK":
        x = psk(8, n, rng)
    elif modulation == "QAM16":
        x = qam(16, n, rng)
    elif modulation == "QAM64":
        x = qam(64, n, rng)
    elif modulation == "BFSK":
        x = bfsk(n, rng)
    elif modulation == "AM-DSB":
        x = am_dsb(n, rng)
    elif modulation == "FM":
        x = fm(n, rng)
    else:
        raise ValueError(f"Unknown modulation: {modulation}")
    return normalize(apply_channel(x, rng))


def stress_signal(x: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    n = len(y)
    t = np.arange(n)
    if condition == "clean":
        return y
    if condition == "low_snr":
        return add_awgn(y, -4, rng)
    if condition == "narrowband_jam":
        tone = np.exp(1j * (2 * np.pi * rng.uniform(0.04, 0.22) * t + rng.uniform(0, 2 * np.pi)))
        return normalize(y + 0.9 * tone)
    if condition == "broadband_jam":
        return add_awgn(y, 0, rng)
    if condition == "frequency_offset":
        return normalize(y * np.exp(1j * 2 * np.pi * 0.045 * t))
    if condition == "multipath":
        z = y.copy()
        for delay, gain in [(2, 0.35), (5, 0.18)]:
            z[delay:] += gain * np.exp(1j * rng.uniform(0, 2 * np.pi)) * y[:-delay]
        return normalize(z)
    if condition == "impulsive_noise":
        mask = rng.random(n) < 0.025
        impulses = 4.5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        z = y.copy()
        z[mask] += impulses[mask]
        return normalize(z)
    raise ValueError(f"Unknown stress condition: {condition}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    snr_values = np.array([-6, 0, 6, 12, 18], dtype=float)
    stress_conditions = [
        "clean",
        "low_snr",
        "narrowband_jam",
        "broadband_jam",
        "frequency_offset",
        "multipath",
        "impulsive_noise",
    ]

    clean_x = []
    clean_y = []
    clean_snr = []
    stress_x = {condition: [] for condition in stress_conditions}
    stress_y = {condition: [] for condition in stress_conditions}

    for label, modulation in enumerate(MODULATIONS):
        for _ in range(args.samples_per_class):
            base = generate_signal(modulation, args.iq_length, rng)
            snr_db = float(rng.choice(snr_values))
            clean = add_awgn(base, snr_db, rng)
            clean_x.append(np.stack([clean.real, clean.imag], axis=0).astype(np.float32))
            clean_y.append(label)
            clean_snr.append(snr_db)
            for condition in stress_conditions:
                z = stress_signal(clean, condition, rng)
                stress_x[condition].append(np.stack([z.real, z.imag], axis=0).astype(np.float32))
                stress_y[condition].append(label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X": np.asarray(clean_x, dtype=np.float32),
        "y": np.asarray(clean_y, dtype=np.int64),
        "snr_db": np.asarray(clean_snr, dtype=np.float32),
        "modulations": np.asarray(MODULATIONS),
        "stress_conditions": np.asarray(stress_conditions),
    }
    for condition in stress_conditions:
        payload[f"X_{condition}"] = np.asarray(stress_x[condition], dtype=np.float32)
        payload[f"y_{condition}"] = np.asarray(stress_y[condition], dtype=np.int64)
    np.savez_compressed(args.out, **payload)
    print(f"Wrote {args.out}")
    print(f"Clean X shape: {payload['X'].shape}")
    print(f"Classes: {MODULATIONS}")
    print(f"Stress conditions: {stress_conditions}")


if __name__ == "__main__":
    main()

