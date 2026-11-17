#!/usr/bin/env python3
"""ApisSonic — train the 18 -> 8 -> 5 hive-state MLP and export weights as JSON.

The feature pipeline mirrors dsp-engine.js / ml-classifier.js exactly:
Blackman window -> |FFT|/N -> 0.75 time smoothing -> dB -> 24 mel bands (60-2000 Hz)
-> orthonormal DCT-II (13 MFCCs) + DominantHz, CentroidHz, Rolloff85Hz, SpectralFlux, ZCR.

Usage
  Synthetic demo (no recordings needed):
      python3 ml/train_hive_classifier.py --synthetic-demo --out ml/custom_mlp_weights.json
  Real recordings (.wav exported from the app, one sub-folder per label):
      python3 ml/train_hive_classifier.py --data-dir recordings/ --out ml/custom_mlp_weights.json
      recordings/healthy/*.wav  recordings/preswarm/*.wav  recordings/queenless/*.wav
      recordings/piping/*.wav   recordings/fanning/*.wav

Input normalisation (mean / std) is folded into W1 and b1, so the app's forward pass
needs nothing beyond W1, b1, W2, b2. Requires numpy.
"""
import argparse
import glob
import json
import os
import sys
import wave

import numpy as np

LABELS = ["healthy", "preswarm", "queenless", "piping", "fanning"]
FFT_SIZE = 4096
SMOOTHING = 0.75
HOP_SECONDS = 0.05
WARMUP_FRAMES = 10
N_MEL, N_MFCC = 24, 13
MEL_LO, MEL_HI = 60.0, 2000.0
N_FEATURES, N_HIDDEN = 18, 8
LEAK = 0.05

_n = np.arange(FFT_SIZE)
WINDOW = 0.42 - 0.5 * np.cos(2 * np.pi * _n / FFT_SIZE) + 0.08 * np.cos(4 * np.pi * _n / FFT_SIZE)


# ----------------------------------------------------------------- features

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr):
    bins = FFT_SIZE // 2
    bw = sr / FFT_SIZE
    edges = mel_to_hz(np.linspace(hz_to_mel(MEL_LO), hz_to_mel(MEL_HI), N_MEL + 2))
    f = np.arange(bins) * bw
    fb = np.zeros((N_MEL, bins))
    for m in range(N_MEL):
        f0, f1, f2 = edges[m], edges[m + 1], edges[m + 2]
        up = (f - f0) / (f1 - f0)
        down = (f2 - f) / (f2 - f1)
        tri = np.where(f <= f1, up, down)
        fb[m] = np.where((f >= f0) & (f <= f2), np.maximum(tri, 0.0), 0.0)
    return fb


_m = np.arange(N_MEL)
DCT = np.array([
    (np.sqrt(1.0 / N_MEL) if c == 0 else np.sqrt(2.0 / N_MEL)) * np.cos(np.pi * c * (_m + 0.5) / N_MEL)
    for c in range(N_MFCC)
])


def clip_features(x, sr):
    """Time signal -> array [frames, 18] using the same framing as the live app."""
    hop = int(round(HOP_SECONDS * sr))
    bw = sr / FFT_SIZE
    fb = mel_filterbank(sr)
    k_lo, k_hi = int(np.ceil(MEL_LO / bw)), int(np.floor(MEL_HI / bw))
    d_lo, d_hi = max(1, int(np.ceil(60 / bw))), int(np.floor(1000 / bw))
    prev_mag = np.zeros(FFT_SIZE // 2)
    prev_norm = None
    feats = []
    i = 0
    pos = FFT_SIZE
    while pos <= len(x):
        frame = x[pos - FFT_SIZE:pos]
        mag = np.abs(np.fft.rfft(frame * WINDOW))[:FFT_SIZE // 2] / FFT_SIZE
        prev_mag = SMOOTHING * prev_mag + (1 - SMOOTHING) * mag
        with np.errstate(divide="ignore"):
            db = 20 * np.log10(prev_mag)
        p = 10.0 ** (db / 10.0)

        e = 10 * np.log10(fb @ p + 1e-10)
        mfcc = DCT @ e
        mfcc[1:] = DCT[1:] @ (e - e.mean())

        k = d_lo + int(np.argmax(db[d_lo:d_hi + 1]))
        a, b, c = db[k - 1], db[k], db[k + 1]
        den = a - 2 * b + c
        delta = 0.0
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and abs(den) > 1e-9:
            delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        dominant = (k + delta) * bw

        seg = p[k_lo:k_hi + 1]
        total = seg.sum()
        kk = np.arange(k_lo, k_hi + 1)
        centroid = float((seg * kk * bw).sum() / total) if total > 0 else 0.0
        cum = np.cumsum(seg)
        rolloff = float(kk[min(int(np.searchsorted(cum, 0.85 * total)), len(kk) - 1)] * bw) if total > 0 else 0.0
        norm = seg / total if total > 0 else seg
        flux = float(np.sqrt(((norm - prev_norm) ** 2).sum())) if prev_norm is not None else 0.0
        prev_norm = norm
        pos_sign = frame >= 0
        zcr = float(np.count_nonzero(pos_sign[1:] != pos_sign[:-1])) * sr / FFT_SIZE

        if i >= WARMUP_FRAMES:
            feats.append(np.concatenate([mfcc, [dominant, centroid, rolloff, flux, zcr]]))
        i += 1
        pos += hop
    return np.array(feats)


# ----------------------------------------------------------------- synthesizer

def pink_noise(n, sr, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    shape = 1.0 / np.sqrt(np.maximum(f, 20.0))
    shape *= f / np.sqrt(f ** 2 + 120.0 ** 2)  # one-pole 120 Hz high-pass
    y = np.fft.irfft(spec * shape, n)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def synthesize(state, seconds, sr, rng, gain=0.2):
    """Same recipes as synthesizeColony() in dsp-engine.js."""
    n = int(round(seconds * sr))
    t = np.arange(n) / sr
    tau = 2 * np.pi
    out = np.zeros(n)
    u = lambda lo, hi: rng.uniform(lo, hi)

    def tone(f0, harm, amp, am_rate=0.0, am_depth=0.0, fm_rate=0.0, fm_dev=0.0, pulse_rate=0.0):
        f = f0 + (fm_dev * np.sin(tau * fm_rate * t + u(0, tau)) if fm_dev else 0.0)
        phase = np.cumsum(np.full(n, tau / sr) * f)
        env = 1.0 + (am_depth * np.sin(tau * am_rate * t + u(0, tau)) if am_depth else 0.0)
        if pulse_rate:
            p = 0.5 + 1.5 * np.sin(tau * pulse_rate * t + u(0, tau))
            env = env * (0.35 + 0.65 * np.clip(p, 0, 1))
        s = sum(h * np.sin((i + 1) * phase + u(0, tau)) for i, h in enumerate(harm))
        return amp * env * s

    if state == "fanning":
        out += tone(u(140, 185), [1, 0.2, 0.06], 1, am_rate=u(3, 5), am_depth=0.25)
        out += 0.1 * pink_noise(n, sr, rng)
    elif state == "preswarm":
        out += tone(u(325, 395), [1, 0.3, 0.1], 1, am_rate=u(1, 2), am_depth=0.2, fm_rate=u(5, 8), fm_dev=8)
        out += tone(u(230, 250), [1], 0.2)
        out += 0.12 * pink_noise(n, sr, rng)
    elif state == "piping":
        out += tone(u(230, 250), [1, 0.2], 0.25)
        out += tone(u(430, 478), [1, 0.7, 0.15], 1, pulse_rate=u(2.5, 4))
        out += 0.08 * pink_noise(n, sr, rng)
    elif state == "queenless":
        out += tone(u(210, 255), [1, 0.2], 0.3, am_rate=u(0.5, 1.2), am_depth=0.1)
        k = 48
        for _ in range(k):
            out += np.sqrt(2.0 / k) * np.sin(tau * u(550, 1000) * t + u(0, tau))
        out += 0.12 * pink_noise(n, sr, rng)
    else:  # healthy
        out += tone(u(225, 265), [1, 0.25, 0.08], 1, am_rate=u(0.6, 1.5), am_depth=0.15)
        out += 0.1 * pink_noise(n, sr, rng)
    return out * gain


def synthetic_dataset(clips_per_class, rng):
    X, y = [], []
    for li, label in enumerate(LABELS):
        for _ in range(clips_per_class):
            sr = int(rng.choice([44100, 48000]))  # desktop browsers and iPhones
            gain = 10 ** (rng.uniform(-40, -8) / 20)  # wide range: mic level is unknown
            f = clip_features(synthesize(label, 1.7, sr, rng, gain), sr)
            X.append(f)
            y.append(np.full(len(f), li))
    return np.vstack(X), np.concatenate(y)


# ----------------------------------------------------------------- real data

def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM is supported")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def wav_dataset(data_dir):
    X, y = [], []
    for li, label in enumerate(LABELS):
        files = sorted(glob.glob(os.path.join(data_dir, label, "*.wav")))
        print(f"  {label}: {len(files)} file(s)")
        for path in files:
            x, sr = read_wav(path)
            # App exports are 2x downsampled (22.05 / 24 kHz); restore the live rate so
            # bin spacing matches what the app sees. Linear interpolation is adequate
            # because all features live below 2 kHz.
            if sr < 40000:
                x = np.interp(np.arange(len(x) * 2) / 2.0, np.arange(len(x)), x)
                sr *= 2
            f = clip_features(x, sr)
            if len(f):
                X.append(f)
                y.append(np.full(len(f), li))
    if not X:
        sys.exit("No .wav files found. Expected <data-dir>/<label>/*.wav")
    return np.vstack(X), np.concatenate(y)


# ----------------------------------------------------------------- training

def train(X, y, rng, epochs=600, lr=0.02, l2=1e-4):
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xn = (X - mean) / std
    n, k = len(Xn), len(LABELS)
    Y = np.eye(k)[y]
    params = {
        "W1": rng.standard_normal((N_HIDDEN, N_FEATURES)) * np.sqrt(2.0 / N_FEATURES),
        "b1": np.zeros(N_HIDDEN),
        "W2": rng.standard_normal((k, N_HIDDEN)) * np.sqrt(2.0 / N_HIDDEN),
        "b2": np.zeros(k),
    }
    m = {key: np.zeros_like(v) for key, v in params.items()}
    v = {key: np.zeros_like(val) for key, val in params.items()}
    for step in range(1, epochs + 1):
        z1 = Xn @ params["W1"].T + params["b1"]
        h = np.maximum(LEAK * z1, z1)
        z2 = h @ params["W2"].T + params["b2"]
        z2 -= z2.max(axis=1, keepdims=True)
        p = np.exp(z2)
        p /= p.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
        d2 = (p - Y) / n
        dh = d2 @ params["W2"]
        d1 = dh * np.where(z1 > 0, 1.0, LEAK)
        grads = {
            "W2": d2.T @ h + l2 * params["W2"],
            "b2": d2.sum(axis=0),
            "W1": d1.T @ Xn + l2 * params["W1"],
            "b1": d1.sum(axis=0),
        }
        for key in params:
            m[key] = 0.9 * m[key] + 0.1 * grads[key]
            v[key] = 0.999 * v[key] + 0.001 * grads[key] ** 2
            params[key] -= lr * (m[key] / (1 - 0.9 ** step)) / (np.sqrt(v[key] / (1 - 0.999 ** step)) + 1e-8)
        if step % 100 == 0 or step == epochs:
            acc = float((p.argmax(axis=1) == y).mean())
            print(f"  epoch {step:4d}  loss {loss:.4f}  train acc {acc * 100:.1f}%")
    # Fold normalisation into the first layer: W1 (x - mean) / std + b1
    W1 = params["W1"] / std
    b1 = params["b1"] - W1 @ mean
    return {"W1": W1, "b1": b1, "W2": params["W2"], "b2": params["b2"]}


def predict(w, X):
    z1 = X @ w["W1"].T + w["b1"]
    h = np.maximum(LEAK * z1, z1)
    return (h @ w["W2"].T + w["b2"]).argmax(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic-demo", action="store_true", help="train on synthesized colony sounds")
    ap.add_argument("--data-dir", help="folder with one sub-folder of .wav files per label")
    ap.add_argument("--out", default="ml/custom_mlp_weights.json")
    ap.add_argument("--clips", type=int, default=90, help="synthetic clips per class")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not args.synthetic_demo and not args.data_dir:
        ap.error("choose --synthetic-demo or --data-dir")
    rng = np.random.default_rng(args.seed)

    if args.data_dir:
        print("Reading recordings …")
        X, y = wav_dataset(args.data_dir)
        source = "recordings"
    else:
        print("Synthesizing training clips …")
        X, y = synthetic_dataset(args.clips, rng)
        source = "synthetic-demo"
    idx = rng.permutation(len(X))
    cut = int(0.85 * len(X))
    tr, va = idx[:cut], idx[cut:]
    print(f"Training on {len(tr)} frames, validating on {len(va)} …")
    w = train(X[tr], y[tr], rng, epochs=args.epochs)
    val_acc = float((predict(w, X[va]) == y[va]).mean())
    print(f"Validation accuracy: {val_acc * 100:.1f}%")

    out = {
        "format": "apissonic-mlp-18-8-5",
        "labels": LABELS,
        "features": [f"MFCC_{i}" for i in range(13)] + ["DominantHz", "CentroidHz", "Rolloff85Hz", "SpectralFlux", "ZCR"],
        "activation": "LeakyReLU(0.05) + softmax",
        "trainedOn": source,
        "validationAccuracy": round(val_acc, 4),
        "W1": np.round(w["W1"], 6).tolist(),
        "b1": np.round(w["b1"], 6).tolist(),
        "W2": np.round(w["W2"], 6).tolist(),
        "b2": np.round(w["b2"], 6).tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
