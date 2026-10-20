"""Offline vocoder measurements.

    tools/.venv/bin/python tools/suite.py --v1
    tools/.venv/bin/python tools/suite.py --plugin AbletonVocoder.jsfx
    tools/.venv/bin/python tools/suite.py --only T1,T8

Tests speak the v2 slider map from SPEC.md. ``--v1`` remaps those onto
AbletonVocoder_v1.jsfx. v1's filter-mode enum is swapped (0 = Retro), and its
band slider is an index into {8,12,16,20,24,28,32}. A test that needs a control
v1 does not have is reported as N/A.

Writes tests/out/report.json and prints a metric table.
Exit 0 when every applicable test passes, 1 when the metric self-check fails,
2 when the plugin fails one or more tests.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signals

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "out"
RENDER = ROOT / "tools" / "build" / "jsfx_render"
BLOCK = 512

V2_BANDS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
V1_BANDS = [8, 12, 16, 20, 24, 28, 32]

# idx, lo, hi, step. Enums and integer sliders use an integer step.
V1_SLIDERS = [
    (1, 0, 3, 1), (2, 0, 6, 1), (3, 50, 500, 1), (4, 5000, 18000, 1),
    (5, 0.1, 100, 0.1), (6, 10, 500, 1), (7, 0, 200, 1), (8, -12, 12, 0.1),
    (9, 0, 100, 1), (10, -24, 24, 0.1), (11, 0, 1, 1), (12, -96, 0, 1),
    (13, 0, 100, 1), (14, 0, 1, 1), (15, 0, 2, 1), (16, 10, 200, 1),
    (17, -96, 0, 1), (18, 0, 1, 1), (19, 0, 1, 1), (20, 0, 2, 1),
    (21, 0, 250, 1), (22, 100, 2000, 1), (23, 10, 400, 1),
]
# Slider 8 (Level) is automated only up to 0 dB. +24 dB of dry white noise
# has large second differences because it is loud, which is not a click.
V2_SLIDERS = [
    (1, 0, 3, 1), (2, 0, 9, 1), (3, 20, 2000, 1), (4, 1000, 20000, 1),
    (5, 10, 200, 1), (6, 0, 1, 1), (7, -96, 0, 1), (8, -24, 0, 0.1),
    (9, 0, 200, 1), (10, 0.1, 100, 0.1), (11, 1, 1000, 1), (12, -12, 12, 0.1),
    (13, 0, 100, 1), (14, 0, 2, 1), (15, 0, 1, 1), (16, -96, 0, 1),
    (17, 0, 100, 1), (18, 0, 1, 1), (19, 0, 1, 0.01), (20, 0, 1, 0.01),
    (21, 0, 3, 1), (22, -24, 24, 1), (23, 20, 1000, 1), (24, 100, 2000, 1),
    (25, 0, 1, 1),
]

# Logical defaults from SPEC section 2, with the test overrides from the brief
# (range 80–18000, 20 bands, Precise, Stereo, and the listed times / percents).
DEFAULTS = {
    "carrier": "noise",
    "bands": 20,
    "low": 80.0,
    "high": 18000.0,
    "bw": 100.0,
    "filter_mode": "precise",
    "gate": -96.0,
    "level": 0.0,
    "depth": 100.0,
    "attack": 5.0,
    "release": 80.0,
    "formant": 0.0,
    "drywet": 100.0,
    "channels": "stereo",
    "enhance": 0,
    "unvoiced": -96.0,
    "sens": 50.0,
    "uv_speed": "slow",
    "noise_rate": 1.0,
    "noise_crackle": 1.0,
    "osc_wave": "saw",
    "osc_pitch": 0,
    "track_low": 60.0,
    "track_high": 1000.0,
    "device_on": 1,
}


class MissingFeature(Exception):
    pass


class RenderError(Exception):
    pass


def metric(name, value, threshold, status, note=""):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        value = None
        if status == "PASS":
            status = "FAIL"
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "status": status,
        "note": note,
    }


def m_le(name, value, limit):
    return metric(name, float(value), f"<= {limit:g}", "PASS" if value <= limit else "FAIL")


def m_ge(name, value, limit):
    return metric(name, float(value), f">= {limit:g}", "PASS" if value >= limit else "FAIL")


def m_abs_le(name, value, limit):
    return metric(name, float(value), f"|x| <= {limit:g}", "PASS" if abs(value) <= limit else "FAIL")


def m_between(name, value, lo, hi):
    ok = lo <= value <= hi
    return metric(name, float(value), f"{lo:g} .. {hi:g}", "PASS" if ok else "FAIL")


def m_info(name, value, note=""):
    if value is None:
        return metric(name, None, "info", "INFO", note)
    return metric(name, float(value), "info", "INFO", note)


def _on(value) -> int:
    if isinstance(value, str):
        return 1 if value.lower() in {"1", "fast", "on", "true", "yes"} else 0
    return 1 if value else 0


def map_sliders(params: dict, v1: bool) -> list[tuple[int, float]]:
    p = dict(DEFAULTS)
    p.update(params)
    if v1:
        if p["bands"] not in V1_BANDS:
            raise MissingFeature(f"v1 bands are {V1_BANDS}, not {p['bands']}")
        if abs(float(p["noise_rate"]) - 1.0) > 1e-9 or abs(float(p["noise_crackle"]) - 1.0) > 1e-9:
            raise MissingFeature("v1 has no Noise Rate / Noise Crackle")
        if p["osc_wave"] not in {"saw", "sine", "square"}:
            raise MissingFeature("v1 oscillator is Saw/Sine/Square only")
        if int(p["osc_pitch"]) != 0:
            raise MissingFeature("v1 has no Osc Pitch")
        if not (50 <= float(p["low"]) <= 500):
            raise MissingFeature(f"v1 min freq range is 50..500, not {p['low']}")
        if not (5000 <= float(p["high"]) <= 18000):
            raise MissingFeature(f"v1 max freq range is 5000..18000, not {p['high']}")
        if not (10 <= float(p["release"]) <= 500):
            raise MissingFeature(f"v1 release range is 10..500 ms, not {p['release']}")
        if not (10 <= float(p["track_low"]) <= 400):
            raise MissingFeature(f"v1 pitch-low range is 10..400 Hz, not {p['track_low']}")
        carrier = {"noise": 0, "external": 1, "modulator": 2, "pitch": 3}[p["carrier"]]
        filt = {"retro": 0, "precise": 1}[p["filter_mode"]]
        ch = {"mono": 0, "stereo": 1, "lr": 2}[p["channels"]]
        wave = {"saw": 0, "sine": 1, "square": 2}[p["osc_wave"]]
        return [
            (1, carrier),
            (2, V1_BANDS.index(p["bands"])),
            (3, float(p["low"])),
            (4, float(p["high"])),
            (5, float(p["attack"])),
            (6, float(p["release"])),
            (7, float(p["depth"])),
            (8, float(p["formant"])),
            (9, float(p["drywet"])),
            (10, float(p["level"])),
            (11, _on(p["enhance"])),
            (12, float(p["unvoiced"])),
            (13, float(p["sens"])),
            (14, _on(p["uv_speed"])),
            (15, wave),
            (16, float(p["bw"])),
            (17, float(p["gate"])),
            (18, _on(p["device_on"])),
            (19, filt),
            (20, ch),
            (21, 0),
            (22, float(p["track_high"])),
            (23, float(p["track_low"])),
        ]
    if p["bands"] not in V2_BANDS:
        raise MissingFeature(f"v2 bands are {V2_BANDS}, not {p['bands']}")
    carrier = {"noise": 0, "external": 1, "modulator": 2, "pitch": 3}[p["carrier"]]
    filt = {"precise": 0, "retro": 1}[p["filter_mode"]]
    ch = {"mono": 0, "stereo": 1, "lr": 2}[p["channels"]]
    wave = {"saw": 0, "pulse50": 1, "pulse25": 2, "pulse12.5": 3, "sine": 1, "square": 2}[p["osc_wave"]]
    return [
        (1, carrier),
        (2, V2_BANDS.index(p["bands"])),
        (3, float(p["low"])),
        (4, float(p["high"])),
        (5, float(p["bw"])),
        (6, filt),
        (7, float(p["gate"])),
        (8, float(p["level"])),
        (9, float(p["depth"])),
        (10, float(p["attack"])),
        (11, float(p["release"])),
        (12, float(p["formant"])),
        (13, float(p["drywet"])),
        (14, ch),
        (15, _on(p["enhance"])),
        (16, float(p["unvoiced"])),
        (17, float(p["sens"])),
        (18, _on(p["uv_speed"])),
        (19, float(p["noise_rate"])),
        (20, float(p["noise_crackle"])),
        (21, wave),
        (22, int(p["osc_pitch"])),
        (23, float(p["track_low"])),
        (24, float(p["track_high"])),
        (25, _on(p["device_on"])),
    ]


def parse_log(text: str) -> dict:
    keys = {
        "wall_s", "rtf", "nan_count", "inf_count", "frames",
        "sample_rate", "peak_abs", "peak_dbfs", "block",
    }
    ints = {"nan_count", "inf_count", "frames", "sample_rate", "block"}
    out = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key not in keys:
            continue
        out[key] = int(float(val)) if key in ints else float(val)
    return out


def render(ctx, audio: np.ndarray, params: dict, tag: str, sr: int | None = None, automate: Path | None = None):
    sr = int(sr or ctx.sr)
    sets = map_sliders(params, ctx.v1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    in_path = OUT_DIR / f"{tag}_in.wav"
    out_path = OUT_DIR / f"{tag}_out.wav"
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = np.stack([data, data, np.zeros_like(data), np.zeros_like(data)], axis=1)
    sf.write(str(in_path), data, sr, subtype="FLOAT")
    cmd = [
        str(ctx.render_bin), str(ctx.plugin), str(in_path), str(out_path),
        "--sr", str(sr), "--block", str(ctx.block),
    ]
    for idx, val in sets:
        cmd.extend(["--set", f"{idx}={val:.10g}"])
    if automate is not None:
        cmd.extend(["--automate", str(automate)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-40:])
        raise RenderError(f"render {tag} failed (exit {proc.returncode})\n{tail}")
    meta = parse_log(proc.stderr)
    y, ysr = sf.read(str(out_path), always_2d=True, dtype="float64")
    if int(ysr) != sr:
        raise RenderError(f"{tag} came back at {ysr} Hz")
    meta["stderr"] = proc.stderr
    return y, meta


def mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def amp_db(x: np.ndarray, floor: float = -240.0) -> float:
    r = rms(x)
    if r <= 1e-20:
        return floor
    return 20.0 * math.log10(r)


def welch_psd(x: np.ndarray, sr: int, nperseg: int = 8192):
    """One-sided power spectral density, Hann window, 50% overlap.

    Implemented with numpy. The installed scipy wheel does not load on this
    OS, and a PSD ratio does not depend on matching scipy's absolute scale.
    """
    x = mono(x)
    nperseg = int(min(nperseg, max(256, len(x) // 4)))
    nperseg = max(256, nperseg)
    if nperseg % 2:
        nperseg += 1
    hop = nperseg // 2
    window = np.hanning(nperseg)
    scale = float(np.sum(window * window)) * sr
    acc = np.zeros(nperseg // 2 + 1)
    count = 0
    for start in range(0, len(x) - nperseg + 1, hop):
        spec = np.fft.rfft(x[start:start + nperseg] * window)
        acc += (np.abs(spec) ** 2) / scale
        count += 1
    if count == 0:
        spec = np.fft.rfft(x * np.hanning(len(x)))
        acc = (np.abs(spec) ** 2) / (float(np.sum(np.hanning(len(x)) ** 2)) * sr)
        freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
        if len(acc) > 2:
            acc[1:-1] *= 2.0
        return freqs, np.maximum(acc, 0.0)
    acc /= count
    acc[1:-1] *= 2.0
    return np.fft.rfftfreq(nperseg, 1.0 / sr), np.maximum(acc, 0.0)


def smooth_mean(freq, values, fraction, fmin, fmax, n_points=128):
    """Mean of `values` inside each 1/fraction-octave window, on a log grid."""
    centers = np.geomspace(fmin, fmax, n_points)
    half = 0.5 / fraction
    out = np.empty(n_points)
    for i, c in enumerate(centers):
        m = (freq >= c / 2.0 ** half) & (freq < c * 2.0 ** half)
        out[i] = float(np.mean(values[m])) if np.any(m) else np.nan
    return centers, out


def band_density_db(x, sr, fmin, fmax, fraction=3.0):
    """Mean PSD in dB inside successive 1/fraction-octave bands.

    Mean density, not integrated band power: a white spectrum then has slope 0,
    which is what |slope| <= 1 dB/oct is asking for. Integrated 1/3-oct power
    would add about +3 dB/oct on white noise.
    """
    f, p = welch_psd(x, sr)
    step = 1.0 / fraction
    n = int(np.floor(np.log2(fmax / fmin) / step))
    centers = fmin * 2.0 ** ((np.arange(n) + 0.5) * step)
    centers = centers[(centers > fmin * 0.98) & (centers < fmax)]
    levels = []
    keep = []
    half = 0.5 / fraction
    for c in centers:
        m = (f >= c / 2.0 ** half) & (f < c * 2.0 ** half)
        if np.count_nonzero(m) < 1:
            continue
        levels.append(10.0 * math.log10(float(np.mean(p[m])) + 1e-30))
        keep.append(c)
    return np.asarray(keep), np.asarray(levels)


def slope_db_per_oct(centers, levels) -> float:
    if len(centers) < 3:
        return float("nan")
    return float(np.polyfit(np.log2(centers), levels, 1)[0])


def ripple_of_ratio(out, carrier, sr, fmin, fmax, fraction) -> tuple[float, float]:
    fo, po = welch_psd(out, sr)
    fc, pc = welch_psd(carrier, sr)
    # Identical welch settings on equal-length signals share the frequency grid.
    n = min(len(po), len(pc))
    ratio = po[:n] / np.maximum(pc[:n], 1e-30)
    centers, sm = smooth_mean(fo[:n], ratio, fraction, fmin, fmax)
    db = 10.0 * np.log10(np.maximum(sm, 1e-30))
    db = db[np.isfinite(db)]
    if len(db) < 4:
        return float("nan"), float("nan")
    return float(np.max(db) - np.min(db)), float(np.median(db))


def distortion_dbc(x, sr, freq, width=15.0, skip_s=0.0) -> float:
    y = mono(x)
    y = y[int(skip_s * sr):]
    if len(y) < 256:
        return float("nan")
    w = np.hanning(len(y))
    mag2 = np.abs(np.fft.rfft(y * w)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    inside = np.abs(freqs - freq) <= width
    e_in = float(np.sum(mag2[inside])) + 1e-30
    e_out = float(np.sum(mag2[~inside]))
    return 10.0 * math.log10(e_out / e_in)


def _parabolic(c, tau: int) -> float:
    if tau <= 0 or tau >= len(c) - 1:
        return float(tau)
    a, b, d = float(c[tau - 1]), float(c[tau]), float(c[tau + 1])
    denom = a - 2.0 * b + d
    if denom == 0.0:
        return float(tau)
    return tau + 0.5 * (a - d) / denom


def yin_f0(frame, sr, fmin, fmax, threshold=0.15) -> float:
    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)
    if float(np.dot(x, x)) < 1e-12:
        return float("nan")
    w = len(x) // 2
    if w < 8:
        return float("nan")
    x = x[: 2 * w]
    tau_min = max(2, int(math.floor(sr / fmax)))
    tau_max = min(w - 2, int(math.ceil(sr / fmin)))
    if tau_max <= tau_min + 2:
        return float("nan")
    sum1 = float(np.sum(x[:w] ** 2))
    c2 = np.cumsum(x ** 2)
    sum2 = c2[w - 1: w - 1 + w] - np.concatenate(([0.0], c2[: w - 1]))
    cross = np.correlate(x, x[:w], mode="valid")[:w]
    d = sum1 + sum2 - 2.0 * cross
    d[0] = 0.0
    d = np.maximum(d, 0.0)
    acc = np.cumsum(d)
    idx = np.arange(w, dtype=np.float64)
    cmnd = np.ones(w)
    nz = acc > 0
    cmnd[nz] = d[nz] * idx[nz] / acc[nz]
    cmnd[0] = 1.0
    region = cmnd[tau_min:tau_max]
    below = np.flatnonzero(region < threshold)
    if len(below) == 0:
        k = int(np.argmin(region))
        if region[k] > 0.40:
            return float("nan")
        tau = tau_min + k
    else:
        tau = tau_min + int(below[0])
        while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
    tau_f = _parabolic(cmnd, tau)
    if tau_f <= 1.0:
        return float("nan")
    return float(sr / tau_f)


def track_f0(y, sr, fmin=50.0, fmax=1000.0, hop_s=0.01, gate_db=-50.0):
    y = mono(y)
    frame = int(round(2.0 * sr / fmin))
    if frame % 2:
        frame += 1
    hop = max(1, int(round(hop_s * sr)))
    times, f0s, levels = [], [], []
    for start in range(0, len(y) - frame + 1, hop):
        seg = y[start:start + frame]
        level = amp_db(seg)
        times.append((start + frame / 2.0) / sr)
        levels.append(level)
        if level < gate_db:
            f0s.append(float("nan"))
        else:
            f0s.append(yin_f0(seg, sr, fmin, fmax))
    return np.asarray(times), np.asarray(f0s), np.asarray(levels)


def envelope_db(x, sr, fmin=400.0, fmax=6000.0, step_st=0.25, fraction=3.0):
    f, p = welch_psd(x, sr, nperseg=4096)
    semis = np.arange(math.log2(fmin) * 12.0, math.log2(fmax) * 12.0, step_st)
    freqs = 2.0 ** (semis / 12.0)
    half = 0.5 / fraction
    db = np.empty(len(freqs))
    for i, c in enumerate(freqs):
        m = (f >= c / 2.0 ** half) & (f < c * 2.0 ** half)
        db[i] = 10.0 * math.log10(float(np.mean(p[m])) + 1e-30) if np.any(m) else np.nan
    return semis, db


def _envelope_shape(env):
    """First difference of a log-frequency envelope.

    Correlating raw levels underestimates a formant shift: a fixed analysis
    window clips a sloping spectrum, so the broadband tilt matches best at a
    lag shorter than the peak displacement. The difference keeps the peaks.
    """
    env = np.asarray(env, dtype=np.float64).copy()
    bad = ~np.isfinite(env)
    if np.any(~bad):
        env[bad] = float(np.nanmean(env))
    return np.diff(env)


def measure_shift_st(env_ref, env_mov, step_st, max_st=18.0) -> tuple[float, float]:
    """Return (shift, correlation). Positive means env_mov sits higher in frequency."""
    ref = _envelope_shape(env_ref)
    mov = _envelope_shape(env_mov)
    ref = ref - np.nanmean(ref)
    mov = mov - np.nanmean(mov)
    max_bins = int(round(max_st / step_st))
    lags = np.arange(-max_bins, max_bins + 1)
    scores = np.full(lags.shape, -np.inf)
    for i, k in enumerate(lags):
        k = int(k)
        if k >= 0:
            a = ref[: len(ref) - k] if k else ref
            b = mov[k:]
        else:
            a = ref[-k:]
            b = mov[: len(mov) + k]
        m = np.isfinite(a) & np.isfinite(b)
        if int(np.count_nonzero(m)) < 8:
            continue
        aa, bb = a[m], b[m]
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom > 0:
            scores[i] = float(np.dot(aa, bb) / denom)
    i = int(np.argmax(scores))
    delta = 0.0
    if 0 < i < len(scores) - 1 and math.isfinite(scores[i - 1] + scores[i + 1]):
        y0, y1, y2 = scores[i - 1], scores[i], scores[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
    return float((lags[i] + delta) * step_st), float(scores[i])


def shift_spectrum(x, semitones):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0)
    # rfftfreq without a sample period is in cycles/sample; the ratio is what matters.
    factor = 2.0 ** (semitones / 12.0)
    src = freqs / factor
    mag = np.interp(src, freqs, np.abs(spec), left=0.0, right=0.0)
    phase = np.interp(src, freqs, np.unwrap(np.angle(spec)), left=0.0, right=0.0)
    return np.fft.irfft(mag * np.exp(1j * phase), n)


def nan_fail(meta) -> list:
    nan = int(meta.get("nan_count", 0))
    inf = int(meta.get("inf_count", 0))
    if nan or inf:
        return [metric("nan_inf_count", nan + inf, "== 0", "FAIL")]
    return []


def quantize(rng, lo, hi, step):
    n = int(round((hi - lo) / step))
    k = int(rng.integers(0, n + 1))
    v = lo + k * step
    if abs(step - round(step)) < 1e-9:
        return int(round(v))
    return float(v)


def slice_to(x, n):
    x = np.asarray(x)
    if len(x) < n:
        raise RuntimeError(f"need {n} samples, have {len(x)}")
    return x[:n]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_t1(ctx):
    car = ctx.cat["white_carrier_10s"]
    mod = ctx.cat["white_mod_10s"]
    audio = signals.assemble(mod, mod, car, car)
    return _test_t1_body(ctx, audio, car, 1.25 * 80.0, 0.8 * 18000.0)


def _test_t1_body(ctx, audio, car, fmin, fmax):
    metrics = []
    renders = {}
    for mode in ("precise", "retro"):
        y, meta = render(ctx, audio, {
            "carrier": "external", "enhance": 0, "filter_mode": mode,
        }, f"t1_{mode}")
        metrics.extend(nan_fail(meta))
        renders[mode] = y
    level = amp_db(renders["precise"])
    metrics.append(m_ge("precise_output_rms_dbfs", level, -60.0))
    for fraction, limit, name in ((3.0, 2.0, "ripple_1_3_oct_db"), (12.0, 6.0, "ripple_1_12_oct_db")):
        rip, med = ripple_of_ratio(renders["precise"], car, ctx.sr, fmin, fmax, fraction)
        metrics.append(m_le(name, rip, limit))
        if fraction == 3.0:
            metrics.append(m_info("precise_ratio_median_db", med))
    # Retro tilt of the same PSD ratio. Positive means the highs are louder.
    fo, po = welch_psd(renders["retro"], ctx.sr)
    fc, pc = welch_psd(car, ctx.sr)
    n = min(len(po), len(pc))
    ratio = po[:n] / np.maximum(pc[:n], 1e-30)
    centers, sm = smooth_mean(fo[:n], ratio, 3.0, fmin, fmax)
    db = 10.0 * np.log10(np.maximum(sm, 1e-30))
    ok = np.isfinite(db)
    tilt = float(np.polyfit(np.log2(centers[ok]), db[ok], 1)[0]) if np.count_nonzero(ok) > 3 else float("nan")
    metrics.append(m_info("retro_tilt_db_per_oct", tilt, "expect positive"))
    return metrics


def test_t2(ctx):
    sr = ctx.sr
    audio = signals.assemble(ctx.cat["sine250"], ctx.cat["sine250"], ctx.cat["sine262"], ctx.cat["sine262"])
    metrics = []
    y, meta = render(ctx, audio, {"carrier": "external", "attack": 5, "release": 80}, "t2_slow")
    metrics.extend(nan_fail(meta))
    dbc = distortion_dbc(y, sr, 262.0, width=15.0, skip_s=0.5)
    metrics.append(m_le("distortion_dbc", dbc, -40.0))
    yf, metaf = render(ctx, audio, {"carrier": "external", "attack": 0.5, "release": 10}, "t2_fast")
    metrics.extend(nan_fail(metaf))
    metrics.append(m_info("distortion_fast_env_dbc", distortion_dbc(yf, sr, 262.0, 15.0, 0.5), "attack 0.5 / release 10"))
    return metrics


def test_t3(ctx):
    sr = ctx.sr
    n = len(ctx.cat["saw"])
    mod = slice_to(ctx.cat["white_mod_10s"], n)
    audio = signals.assemble(mod, mod, ctx.cat["saw"], ctx.cat["saw"])
    metrics = []
    slopes = {}
    for enh, tag in ((0, "t3_off"), (1, "t3_on")):
        y, meta = render(ctx, audio, {"carrier": "external", "enhance": enh}, tag)
        metrics.extend(nan_fail(meta))
        c, lev = band_density_db(y, sr, 200.0, 10000.0, 3.0)
        slopes[enh] = slope_db_per_oct(c, lev)
    metrics.append(m_info("slope_off_db_per_oct", slopes[0]))
    metrics.append(m_abs_le("slope_on_db_per_oct", slopes[1], 1.0))
    metrics.append(m_ge("slope_improvement_db_per_oct", slopes[1] - slopes[0], 3.0))
    return metrics


def test_t4(ctx):
    sr = ctx.sr
    speech = signals._scale_rms(ctx.cat["speech"], -18.0)
    n = min(len(speech), int(5.0 * sr))
    speech = speech[:n]
    audio = signals.assemble(speech, speech, speech, speech)
    y, meta = render(ctx, audio, {
        "carrier": "modulator", "enhance": 1, "depth": 100, "drywet": 100, "level": 0,
    }, "t4")
    metrics = nan_fail(meta)
    metrics.append(m_abs_le("rms_delta_db", amp_db(y) - amp_db(speech), 1.5))
    cin, lin = band_density_db(speech, sr, 100.0, 8000.0, 3.0)
    cout, lout = band_density_db(y, sr, 100.0, 8000.0, 3.0)
    # Match on frequency. Same grid because both calls use the same fmin/fmax/sr.
    m = min(len(lin), len(lout))
    diff = lout[:m] - lin[:m]
    peak = float(np.max(lin[:m]))
    keep = lin[:m] > peak - 40.0
    std = float(np.std(diff[keep])) if np.count_nonzero(keep) > 3 else float("nan")
    metrics.append(metric(
        "spectrum_diff_std_db", std, "<= 2.5",
        "PASS" if std <= 2.5 else "FAIL",
        "1/3-oct mean-PSD, 100 Hz–8 kHz, bands within 40 dB of the input peak",
    ))
    return metrics


def test_t5(ctx):
    sr = ctx.sr
    speech = signals._scale_rms(ctx.cat["speech"], -18.0)
    n = min(len(speech), len(ctx.cat["chord"]), int(2.5 * sr))
    audio = signals.assemble(speech[:n], speech[:n], ctx.cat["chord"][:n], ctx.cat["chord"][:n])
    counts = V1_BANDS if ctx.v1 else V2_BANDS
    levels = []
    metrics = []
    for b in counts:
        y, meta = render(ctx, audio, {"carrier": "external", "bands": b}, f"t5_b{b}")
        metrics.extend(nan_fail(meta))
        # Skip the filter settle so band-count switches aren't dominated by the onset.
        level = amp_db(y[int(0.25 * sr):])
        levels.append(level)
        metrics.append(m_info(f"rms_dbfs_{b}bands", level))
    spread = float(max(levels) - min(levels)) if levels else float("nan")
    metrics.append(m_le("rms_spread_db", spread, 2.0))
    return metrics


def test_t6(ctx):
    sr = ctx.sr
    floor = ctx.cat["floor"]
    n = len(floor["audio"])
    rng = np.random.default_rng(7)
    car = signals.white(n, rng, rms_db=-12.0)
    audio = signals.assemble(floor["audio"], floor["audio"], car, car)
    metrics = []
    renders = {}
    for gate, tag in ((-40.0, "t6_gate"), (-96.0, "t6_open")):
        y, meta = render(ctx, audio, {"carrier": "external", "gate": gate, "enhance": 0, "unvoiced": -96}, tag)
        metrics.extend(nan_fail(meta))
        renders[tag] = y

    def region_db(y, spans):
        chunks = [mono(y)[a:b] for a, b in spans if b > a]
        if not chunks:
            return float("nan")
        return amp_db(np.concatenate(chunks))

    pause = region_db(renders["t6_gate"], floor["pauses"])
    speech_on = region_db(renders["t6_gate"], floor["speech"])
    speech_off = region_db(renders["t6_open"], floor["speech"])
    metrics.append(m_le("pause_rms_dbfs", pause, -80.0))
    metrics.append(m_abs_le("speech_vs_gate_off_db", speech_on - speech_off, 3.0))
    metrics.append(m_info("speech_gate_on_dbfs", speech_on))
    metrics.append(m_info("speech_gate_off_dbfs", speech_off))
    return metrics


def _frame_powers(x, sr, frame, cutoff=4000.0):
    """Per-frame (hf_power, total_power, rms_db) summed across channels."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    w = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    hf_m = freqs >= cutoff
    rows = []
    for start in range(0, x.shape[0] - frame + 1, frame):
        hf = 0.0
        tot = 0.0
        acc = 0.0
        n = 0
        for c in range(x.shape[1]):
            seg = x[start:start + frame, c]
            spec = np.abs(np.fft.rfft(seg * w)) ** 2
            hf += float(np.sum(spec[hf_m]))
            tot += float(np.sum(spec))
            acc += float(np.dot(seg, seg))
            n += frame
        rows.append((hf, tot + 1e-30, 10.0 * math.log10(math.sqrt(acc / n) + 1e-20)))
    return rows


def test_t7(ctx):
    sr = ctx.sr
    speech = signals._scale_rms(ctx.cat["speech"], -18.0)
    n = min(len(speech), len(ctx.cat["chord"]))
    speech = speech[:n]
    chord = ctx.cat["chord"][:n]
    audio = signals.assemble(speech, speech, chord, chord)
    frame = int(round(0.020 * sr))
    # Classify the modulator once.
    mono_rows = _frame_powers(speech, sr, frame)
    fric, voiced = [], []
    for i, (hf, tot, level) in enumerate(mono_rows):
        if level < -40.0:
            continue
        frac = hf / tot
        if frac > 0.5:
            fric.append(i)
        elif frac < 0.1:
            voiced.append(i)
    metrics = [
        m_info("fricative_frames", len(fric)),
        m_info("voiced_frames", len(voiced)),
    ]
    if len(fric) < 5 or len(voiced) < 5:
        metrics.append(metric("classifier", None, ">= 5 frames each", "FAIL", "not enough fricative/voiced frames"))
        return metrics

    def run(tag, **params):
        y, meta = render(ctx, audio, dict({
            "carrier": "external", "enhance": 0,
        }, **params), tag)
        return y, meta

    off, meta = run("t7_off", unvoiced=-96, sens=50)
    s50, meta50 = run("t7_s50", unvoiced=0, sens=50)
    s0, meta0 = run("t7_s0", unvoiced=0, sens=0)
    s100, meta100 = run("t7_s100", unvoiced=0, sens=100)
    for m in (meta, meta50, meta0, meta100):
        metrics.extend(nan_fail(m))

    def hf_db(y, indices):
        rows = _frame_powers(y, sr, frame)
        powers = [rows[i][0] for i in indices]
        return 10.0 * math.log10(float(np.mean(powers)) + 1e-30)

    fric_on = hf_db(s50, fric)
    fric_off = hf_db(off, fric)
    voi_on = hf_db(s50, voiced)
    voi_off = hf_db(off, voiced)
    metrics.append(m_ge("fricative_hf_improvement_db", fric_on - fric_off, 10.0))
    metrics.append(m_abs_le("voiced_hf_change_db", voi_on - voi_off, 3.0))
    contrast_delta = (fric_on - voi_on) - (fric_off - voi_off)
    metrics.append(m_info("contrast_delta_db", contrast_delta))
    residual = amp_db(s0 - off) - amp_db(off)
    metrics.append(m_le("sens0_residual_db", residual, -60.0))
    voi_100 = hf_db(s100, voiced)
    metrics.append(m_ge("sens100_voiced_hf_increase_db", voi_100 - voi_off, 6.0))
    return metrics


def _pitch_report(est, truth_f0, truth_voiced, times, sr, frame_len):
    """est is per analysis frame. truth_* are per sample."""
    hop_centers = times
    gt = np.interp(hop_centers, np.arange(len(truth_f0)) / sr, truth_f0)
    # A frame counts as voiced when its middle half is voiced in the ground truth.
    half = frame_len // 4
    voiced = []
    for t in hop_centers:
        c = int(round(t * sr))
        a = max(0, c - half)
        b = min(len(truth_voiced), c + half)
        voiced.append(bool(np.mean(truth_voiced[a:b]) > 0.9) and gt[len(voiced)] > 0)
    voiced = np.asarray(voiced)
    gt = np.where(voiced, gt, np.nan)
    cents = np.full(len(est), np.nan)
    good = voiced & np.isfinite(est) & (est > 0)
    cents[good] = 1200.0 * np.log2(est[good] / gt[good])
    octave = voiced & (~good | (np.abs(cents) > 600.0))
    n_voiced = int(np.count_nonzero(voiced))
    if n_voiced == 0 or not np.any(np.isfinite(cents)):
        return float("inf"), 100.0, n_voiced
    med = float(np.median(np.abs(cents[np.isfinite(cents)])))
    oct_pct = 100.0 * float(np.count_nonzero(octave)) / n_voiced
    return med, oct_pct, n_voiced


def test_t8(ctx):
    sr = ctx.sr
    voice = ctx.cat["voice"]
    audio = signals.assemble(voice["audio"], voice["audio"], np.zeros_like(voice["audio"]), np.zeros_like(voice["audio"]))
    y, meta = render(ctx, audio, {
        "carrier": "pitch", "osc_wave": "saw", "track_low": 60, "track_high": 1000,
        "enhance": 0, "unvoiced": -96,
    }, "t8_synth")
    metrics = nan_fail(meta)
    fmin, fmax = 50.0, 1200.0
    # The carrier can be a clean pitch well below -50 dBFS. Gate only
    # numerical silence; voiced/unvoiced comes from the ground truth.
    times, est, _levels = track_f0(y, sr, fmin=fmin, fmax=fmax, hop_s=0.01, gate_db=-80.0)
    frame_len = int(round(2.0 * sr / fmin))
    med, oct_pct, n_voiced = _pitch_report(est, voice["f0"], voice["voiced"], times, sr, frame_len)
    metrics.append(m_info("voiced_frames", n_voiced))
    metrics.append(m_le("median_abs_cents", med, 25.0))
    metrics.append(m_le("octave_error_pct", oct_pct, 2.0))

    # Silent gaps: the oscillator should keep the last pitch. Report only.
    gap_cents = []
    gap_levels = []
    y_m = mono(y)
    for a, b in voice["gaps"]:
        prev = np.flatnonzero(voice["voiced"][:a])
        if len(prev) == 0 or b - a < int(0.12 * sr):
            continue
        held = float(voice["f0"][prev[-1]])
        s = a + int(0.05 * sr)
        e = b - int(0.02 * sr)
        if e <= s:
            continue
        gap_levels.append(amp_db(y_m[s:e]))
        # Frames whose centers lie in the gap.
        sel = (times >= s / sr) & (times <= e / sr)
        ee = est[sel]
        ee = ee[np.isfinite(ee) & (ee > 0)]
        if len(ee):
            gap_cents.append(1200.0 * np.log2(ee / held))
    if gap_levels:
        metrics.append(m_info("gap_rms_dbfs", float(np.median(gap_levels))))
    if gap_cents:
        pooled = np.concatenate(gap_cents)
        metrics.append(m_info("gap_hold_median_abs_cents", float(np.median(np.abs(pooled)))))
    else:
        metrics.append(m_info("gap_hold_median_abs_cents", None, "no measurable pitch in the gaps"))

    speech = signals._scale_rms(ctx.cat["speech"], -18.0)
    n = min(len(speech), int(4.0 * sr))
    sp = speech[:n]
    ys, metas = render(ctx, signals.assemble(sp, sp, np.zeros(n), np.zeros(n)), {
        "carrier": "pitch", "osc_wave": "saw", "track_low": 60, "track_high": 1000,
        "enhance": 0, "unvoiced": -96,
    }, "t8_speech")
    metrics.extend(nan_fail(metas))
    _t, fest, _lv = track_f0(ys, sr, fmin=fmin, fmax=fmax, hop_s=0.01, gate_db=-80.0)
    good = np.isfinite(fest) & (fest > 0)
    jumps = 0
    pairs = 0
    idx = np.flatnonzero(good)
    for i, j in zip(idx[:-1], idx[1:]):
        if j != i + 1:
            continue
        pairs += 1
        cents = abs(1200.0 * math.log2(fest[j] / fest[i]))
        if cents > 600.0:
            jumps += 1
    frac = (100.0 * jumps / pairs) if pairs else float("nan")
    metrics.append(m_info("speech_jump_gt_600_pct", frac))
    return metrics


def test_t9(ctx):
    sr = ctx.sr
    speech = signals._scale_rms(ctx.cat["speech"], -18.0)
    n = min(len(speech), int(3.0 * sr))
    speech = speech[:n]
    left, right = signals.decorr_stereo(n, 1004, 1005, rms_db=-12.0)
    audio = signals.assemble(speech, np.zeros(n), left, right)
    metrics = []

    def active_mask(hop=None):
        frame = int(0.02 * sr)
        mask = np.zeros(n, dtype=bool)
        for s in range(0, n - frame + 1, frame):
            if amp_db(speech[s:s + frame]) > -30.0:
                mask[s:s + frame] = True
        return mask

    mask = active_mask()

    y_lr, meta = render(ctx, audio, {"carrier": "external", "channels": "lr", "enhance": 0}, "t9_lr")
    metrics.extend(nan_fail(meta))
    if np.any(mask):
        metrics.append(m_le("lr_right_rms_dbfs", amp_db(y_lr[mask, 1]), -80.0))
        metrics.append(m_ge("lr_left_rms_dbfs", amp_db(y_lr[mask, 0]), -40.0))
    else:
        metrics.append(metric("lr_active", None, "speech present", "FAIL"))

    y_st, meta = render(ctx, audio, {"carrier": "external", "channels": "stereo", "enhance": 0}, "t9_st")
    metrics.extend(nan_fail(meta))
    metrics.append(m_ge("stereo_l_rms_dbfs", amp_db(y_st[:, 0]), -40.0))
    metrics.append(m_ge("stereo_r_rms_dbfs", amp_db(y_st[:, 1]), -40.0))
    a = y_st[:, 0] - np.mean(y_st[:, 0])
    b = y_st[:, 1] - np.mean(y_st[:, 1])
    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b))) + 1e-30
    corr = float(np.dot(a, b) / denom)
    metrics.append(m_le("stereo_lr_correlation", abs(corr), 0.5))
    metrics.append(m_info("stereo_lr_correlation_signed", corr))

    y_mo, meta = render(ctx, audio, {"carrier": "external", "channels": "mono", "enhance": 0}, "t9_mo")
    metrics.extend(nan_fail(meta))
    diff = float(np.max(np.abs(y_mo[:, 0] - y_mo[:, 1])))
    metrics.append(m_le("mono_max_abs_diff", diff, 1e-6))
    return metrics


def test_t10(ctx):
    metrics = []
    worst_silence = 0.0
    silence_bad = 0
    worst_square = -240.0
    square_bad = 0
    for sr in (44100, 48000, 96000):
        n = int(1.5 * sr)
        z = np.zeros((n, 4), dtype=np.float64)
        y, meta = render(ctx, z, {}, f"t10_sil_{sr}", sr=sr)
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        worst_silence = max(worst_silence, peak)
        silence_bad += int(meta.get("nan_count", 0) + meta.get("inf_count", 0))
        metrics.append(m_info(f"silence_peak_{sr}", peak))
        sq = signals.square(int(0.5 * sr), sr, 440.0, amp=1.0)
        audio = np.stack([sq, sq, sq, sq], axis=1)
        _yq, meta = render(ctx, audio, {}, f"t10_sq_{sr}", sr=sr)
        pk = float(meta.get("peak_dbfs", 99.0))
        worst_square = max(worst_square, pk)
        square_bad += int(meta.get("nan_count", 0) + meta.get("inf_count", 0))
        metrics.append(m_info(f"square_peak_dbfs_{sr}", pk))
    metrics.append(m_le("silence_peak_abs", worst_silence, 1e-9))
    metrics.append(metric("silence_nan_inf", silence_bad, "== 0", "PASS" if silence_bad == 0 else "FAIL"))
    metrics.append(m_le("square_peak_dbfs", worst_square, 12.0))
    metrics.append(metric("square_nan_inf", square_bad, "== 0", "PASS" if square_bad == 0 else "FAIL"))

    sr = 48000
    n = int(20.0 * sr)
    chs = [signals.white(n, np.random.default_rng(s), rms_db=-18.0) for s in (42, 43, 44, 45)]
    audio = np.stack(chs, axis=1)
    specs = V1_SLIDERS if ctx.v1 else V2_SLIDERS
    rng = np.random.default_rng(20261020)
    lines = ["sample_offset,slider_idx,value"]
    pos = 2048
    while pos < n:
        for idx, lo, hi, step in specs:
            lines.append(f"{pos},{idx},{quantize(rng, lo, hi, step):.10g}")
        pos += 2048
    csv_path = OUT_DIR / "t10_automate.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n")

    y_ref, meta_ref = render(ctx, audio, {}, "t10_ref", sr=sr)
    y_auto, meta_auto = render(ctx, audio, {}, "t10_auto", sr=sr, automate=csv_path)
    metrics.extend(nan_fail(meta_auto))
    metrics.append(m_le("auto_peak_dbfs", float(meta_auto.get("peak_dbfs", 99)), 12.0))
    d_ref = float(np.max(np.abs(np.diff(y_ref, n=2, axis=0))))
    d_auto = float(np.max(np.abs(np.diff(y_auto, n=2, axis=0))))
    ratio = d_auto / max(d_ref, 1e-12)
    metrics.append(m_info("click_ref", d_ref))
    metrics.append(m_info("click_auto", d_auto))
    metrics.append(m_le("click_ratio", ratio, 4.0))
    return metrics


def test_t11a(ctx):
    sr = ctx.sr
    n = min(len(ctx.cat["chord"]), int(4.0 * sr))
    speech = signals._scale_rms(ctx.cat["speech"], -18.0)[:n]
    chord = ctx.cat["chord"][:n]
    audio = signals.assemble(speech, speech, chord, chord)
    _y, meta = render(ctx, audio, {
        "carrier": "external", "bands": 20, "channels": "stereo",
        "enhance": 0, "unvoiced": -96,
    }, "t11a")
    metrics = nan_fail(meta)
    rtf = float(meta.get("rtf", 0.0))
    metrics.append(m_ge("realtime_factor", rtf, 10.0))
    metrics.append(m_info("wall_s", float(meta.get("wall_s", float("nan")))))
    metrics.append(m_info("bands", 20))
    return metrics


def test_t11b(ctx):
    sr = ctx.sr
    voice = ctx.cat["voice"]["audio"]
    n = min(len(voice), int(4.0 * sr))
    x = voice[:n]
    audio = signals.assemble(x, x, np.zeros(n), np.zeros(n))
    bands = 32 if ctx.v1 else 40
    _y, meta = render(ctx, audio, {
        "carrier": "pitch", "bands": bands, "channels": "lr",
        "enhance": 1, "unvoiced": 0, "osc_wave": "saw",
        "track_low": 60, "track_high": 1000,
    }, "t11b")
    metrics = nan_fail(meta)
    rtf = float(meta.get("rtf", 0.0))
    metrics.append(m_ge("realtime_factor", rtf, 2.0))
    metrics.append(m_info("wall_s", float(meta.get("wall_s", float("nan")))))
    metrics.append(m_info("bands", bands))
    return metrics


def test_t12(ctx):
    sr = ctx.sr
    speech = signals._scale_rms(ctx.cat["speech"], -18.0)
    n = min(len(speech), int(5.0 * sr))
    speech = speech[:n]
    noise = signals.white(n, np.random.default_rng(1201), rms_db=-18.0)
    ext = signals.assemble(speech, speech, noise, noise)
    base = {"carrier": "external", "enhance": 0, "depth": 100, "drywet": 100}
    y0, m0 = render(ctx, ext, dict(base, formant=0), "t12_ext0")
    y7, m7 = render(ctx, ext, dict(base, formant=7), "t12_ext7")
    ym, mm = render(ctx, ext, dict(base, formant=-7), "t12_extm7")
    metrics = nan_fail(m0) + nan_fail(m7) + nan_fail(mm)
    _s0, e0 = envelope_db(y0, sr)
    _s7, e7 = envelope_db(y7, sr)
    _sm, em = envelope_db(ym, sr)
    shift, corr = measure_shift_st(e0, e7, 0.25)
    shift_dn, corr_dn = measure_shift_st(e0, em, 0.25)
    metrics.append(m_between("formant_shift_st", shift, 7.0 - 1.5, 7.0 + 1.5))
    metrics.append(m_between("formant_shift_down_st", shift_dn, -7.0 - 1.5, -7.0 + 1.5))
    metrics.append(m_info("formant_corr", corr))
    metrics.append(m_info("formant_corr_down", corr_dn))

    # Carrier=Modulator + Enhance. The excitation is flat, so the envelope
    # follows the shifted synthesis centres. Wider tolerance than the
    # external measurement: one noise draw moves the downward lag a little.
    mod = signals.assemble(speech, speech, speech, speech)
    base_m = {"carrier": "modulator", "enhance": 1, "depth": 100, "drywet": 100}
    ym0, mm0 = render(ctx, mod, dict(base_m, formant=0), "t12_mod0")
    ym7, mm7 = render(ctx, mod, dict(base_m, formant=7), "t12_mod7")
    ymm, mmm = render(ctx, mod, dict(base_m, formant=-7), "t12_modm7")
    metrics.extend(nan_fail(mm0))
    metrics.extend(nan_fail(mm7))
    metrics.extend(nan_fail(mmm))
    _a, em0 = envelope_db(ym0, sr)
    _b, em7 = envelope_db(ym7, sr)
    _c, emm = envelope_db(ymm, sr)
    shift_m, corr_m = measure_shift_st(em0, em7, 0.25)
    shift_md, corr_md = measure_shift_st(em0, emm, 0.25)
    metrics.append(m_between("formant_shift_modulator_st", shift_m, 7.0 - 2.0, 7.0 + 2.0))
    metrics.append(m_between("formant_shift_modulator_down_st", shift_md, -7.0 - 2.0, -7.0 + 2.0))
    metrics.append(m_info("formant_corr_modulator", corr_m))
    metrics.append(m_info("formant_corr_modulator_down", corr_md))
    return metrics


def test_t13(ctx):
    sr = ctx.sr
    n = int(4.0 * sr)
    mod = signals.white(n, np.random.default_rng(1301), rms_db=-12.0)
    # Carrier pins are unused in Noise mode; keep them silent.
    audio = signals.assemble(mod, mod, np.zeros(n), np.zeros(n))
    base = {"carrier": "noise", "enhance": 0, "unvoiced": -96}
    # Raises MissingFeature on v1 before any render.
    map_sliders(dict(base, noise_rate=0.3, noise_crackle=1.0), ctx.v1)
    y_slow, m1 = render(ctx, audio, dict(base, noise_rate=0.3, noise_crackle=1.0), "t13_rate")
    y_full, m2 = render(ctx, audio, dict(base, noise_rate=1.0, noise_crackle=1.0), "t13_full")
    y_cr, m3 = render(ctx, audio, dict(base, noise_rate=1.0, noise_crackle=0.2), "t13_crackle")
    metrics = nan_fail(m1) + nan_fail(m2) + nan_fail(m3)

    def centroid(x):
        f, p = welch_psd(x, sr, nperseg=4096)
        return float(np.sum(f * p) / (np.sum(p) + 1e-30))

    def hf_share_db(x, cutoff=8000.0):
        f, p = welch_psd(x, sr, nperseg=4096)
        df = float(f[1] - f[0]) if len(f) > 1 else 1.0
        tot = float(np.sum(p) * df) + 1e-30
        hf = float(np.sum(p[f >= cutoff]) * df)
        return 10.0 * math.log10(hf / tot)

    c_slow, c_full = centroid(y_slow), centroid(y_full)
    metrics.append(m_info("centroid_hz_rate_0.3", c_slow))
    metrics.append(m_info("centroid_hz_rate_1", c_full))
    metrics.append(m_info("centroid_ratio", c_slow / max(c_full, 1e-9)))
    metrics.append(m_info("hf_share_db_rate_0.3", hf_share_db(y_slow)))
    metrics.append(m_info("hf_share_db_rate_1", hf_share_db(y_full)))

    def crest(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[int(0.2 * sr):]
        peak = float(np.max(np.abs(x))) + 1e-20
        return 20.0 * math.log10(peak / max(rms(x), 1e-20))

    c_crack = crest(y_cr)
    c_smooth = crest(y_full)
    metrics.append(m_info("crest_db_crackle_0.2", c_crack))
    metrics.append(m_info("crest_db_crackle_1", c_smooth))
    metrics.append(m_ge("crest_increase_db", c_crack - c_smooth, 6.0))
    return metrics


def _active_rms_db(y, sr, skip_s=0.25):
    s = int(skip_s * sr)
    return amp_db(y[s:] if len(y) > s + 8 else y)


def test_t14(ctx):
    """Output level for a -18 dBFS speech modulator, Enhance off and on."""
    sr = ctx.sr
    speech = signals._scale_rms(ctx.cat["speech"], -18.0)
    n = min(len(speech), len(ctx.cat["chord"]), int(4.0 * sr))
    speech = speech[:n]
    chord = signals._scale_rms(ctx.cat["chord"][:n], -18.0)
    z = np.zeros(n)
    ref = _active_rms_db(speech, sr)
    cases = [
        ("noise", signals.assemble(speech, speech, z, z), {"carrier": "noise"}),
        ("pitch", signals.assemble(speech, speech, z, z), {"carrier": "pitch", "osc_wave": "saw"}),
        ("external", signals.assemble(speech, speech, chord, chord), {"carrier": "external"}),
        ("modulator", signals.assemble(speech, speech, speech, speech), {"carrier": "modulator"}),
    ]
    metrics = [m_info("input_rms_dbfs", ref)]
    for name, audio, params in cases:
        levels = {}
        for enh in (0, 1):
            y, meta = render(ctx, audio, dict(params, enhance=enh, unvoiced=-96), f"t14_{name}_{enh}")
            metrics.extend(nan_fail(meta))
            level = _active_rms_db(y, sr)
            levels[enh] = level
            metrics.append(m_info(f"{name}_enh{enh}_rms_dbfs", level))
            metrics.append(m_abs_le(f"{name}_enh{enh}_delta_db", level - ref, 6.0))
        metrics.append(m_abs_le(f"{name}_enhance_delta_db", levels[1] - levels[0], 6.0))
    return metrics


def _rms_envelope(x, sr, win_s):
    x = mono(x)
    w = max(1, int(round(win_s * sr)))
    power = np.convolve(x * x, np.ones(w) / w, mode="same")
    return np.sqrt(np.maximum(power, 0.0))


def _fall_time_60db(env, sr, stop, pre):
    """Seconds after `stop` until the envelope is 60 dB under `pre`.

    A 300 ms release reaches that point at 2.07 s, 70 ms after a 2 s tail.
    When the buffer ends first, extrapolate the log-envelope slope.
    """
    thr = pre * 10.0 ** (-60.0 / 20.0)
    after = env[stop:]
    hit = np.flatnonzero(after <= thr)
    if len(hit):
        return float(hit[0] / sr)
    t0 = int(0.40 * sr)
    t1 = max(t0 + 8, len(after) - int(0.02 * sr))
    seg = after[t0:t1]
    keep = seg > max(thr * 0.01, 1e-12)
    if int(np.count_nonzero(keep)) < 16:
        return float("inf")
    tt = np.arange(len(seg))[keep] / sr
    slope, intercept = np.polyfit(tt, np.log(seg[keep]), 1)
    if slope >= -1e-8:
        return float("inf")
    return float(t0 / sr + (math.log(max(thr, 1e-30)) - intercept) / slope)


def test_t15(ctx):
    sr = ctx.sr
    n_noise = int(1.0 * sr)
    n_sil = int(2.0 * sr)
    pink = signals.pink(n_noise, np.random.default_rng(1501), rms_db=-18.0)
    mod = np.concatenate([pink, np.zeros(n_sil)])
    white = signals.white(len(mod), np.random.default_rng(1502), rms_db=-18.0)
    ext = signals.assemble(mod, mod, white, white)
    metrics = []
    for rel_ms in (80.0, 300.0):
        y, meta = render(ctx, ext, {
            "carrier": "external", "release": rel_ms, "attack": 5, "enhance": 0, "unvoiced": -96,
        }, f"t15_rel{int(rel_ms)}")
        metrics.extend(nan_fail(meta))
        env = _rms_envelope(y, sr, 0.005)
        pre = float(np.median(env[int(0.40 * sr):n_noise - int(0.05 * sr)]))
        got = _fall_time_60db(env, sr, n_noise, pre)
        target = 6.9 * rel_ms * 0.001
        metrics.append(m_info(f"pre_dbfs_rel{int(rel_ms)}", 20.0 * math.log10(pre + 1e-20)))
        metrics.append(m_between(f"fall_60db_s_rel{int(rel_ms)}", got, target * 0.7, target * 1.3))
    # Late window must be silent for every carrier. 7*6.9*80 ms is past the
    # buffer, so the check starts at the 1.9 s cap.
    offset = min(7.0 * 6.9 * 0.080, 1.9)
    start = n_noise + int(offset * sr)
    z = np.zeros(len(mod))
    modes = [
        ("noise", signals.assemble(mod, mod, z, z), {"carrier": "noise"}),
        ("pitch", signals.assemble(mod, mod, z, z), {"carrier": "pitch", "osc_wave": "saw"}),
        ("external", ext, {"carrier": "external"}),
        ("modulator", signals.assemble(mod, mod, mod, mod), {"carrier": "modulator"}),
    ]
    for name, audio, params in modes:
        y, meta = render(ctx, audio, dict(params, release=80, attack=5, enhance=0, unvoiced=-96), f"t15_floor_{name}")
        metrics.extend(nan_fail(meta))
        metrics.append(m_le(f"late_rms_dbfs_{name}", amp_db(y[start:]), -100.0))
    return metrics


def test_t16(ctx):
    sr = ctx.sr
    n_sil = int(0.50 * sr)
    n_burst = int(1.0 * sr)
    pink = signals.pink(n_burst, np.random.default_rng(1601), rms_db=-18.0)
    mod = np.concatenate([np.zeros(n_sil), pink])
    white = signals.white(len(mod), np.random.default_rng(1602), rms_db=-18.0)
    audio = signals.assemble(mod, mod, white, white)
    metrics = []
    rises = {}
    for attack in (1.0, 30.0):
        y, meta = render(ctx, audio, {
            "carrier": "external", "attack": attack, "release": 80, "enhance": 0, "unvoiced": -96,
        }, f"t16_a{int(attack)}")
        metrics.extend(nan_fail(meta))
        env = _rms_envelope(y, sr, 0.005)
        settled = float(np.median(env[n_sil + int(0.60 * sr):]))
        lo, hi = 0.10 * settled, 0.90 * settled
        seg = env[n_sil:]
        above_lo = np.flatnonzero(seg >= lo)
        above_hi = np.flatnonzero(seg >= hi)
        if len(above_lo) == 0 or len(above_hi) == 0:
            rise = float("inf")
        else:
            t_lo = int(above_lo[0])
            later = above_hi[above_hi >= t_lo]
            rise = float("inf") if len(later) == 0 else float((int(later[0]) - t_lo) / sr)
        rises[attack] = rise
        metrics.append(m_info(f"rise_10_90_s_attack{int(attack)}", rise))
    ratio = rises[30.0] / rises[1.0] if rises[1.0] > 0 else float("inf")
    metrics.append(m_ge("rise_ratio_30_over_1", ratio, 5.0))
    return metrics


def validate_clean(ctx) -> list:
    metrics = []
    sr = 48000
    n = int(2.0 * sr)
    pure = signals.sine(n, sr, 262.0, -12.0)
    metrics.append(m_le("self_pure_sine_dbc", distortion_dbc(pure, sr, 262.0), -45.0))
    harm = pure + 0.1 * signals.sine(n, sr, 524.0, -12.0)
    metrics.append(m_between("self_harmonic_dbc", distortion_dbc(harm, sr, 262.0), -25.0, -15.0))

    centers = np.geomspace(200.0, 10000.0, 24)
    levels = -6.0 * np.log2(centers / centers[0])
    slope = slope_db_per_oct(centers, levels)
    metrics.append(m_abs_le("self_slope_error_db_per_oct", slope + 6.0, 0.05))

    # Flat ratio → zero ripple. Known +3 dB/oct tilt comes back as +3.
    freq = np.linspace(0.0, sr / 2.0, 8193)[1:]
    flat = np.ones_like(freq)
    c, sm = smooth_mean(freq, flat, 3.0, 100.0, 14400.0)
    db = 10.0 * np.log10(np.maximum(sm, 1e-30))
    metrics.append(m_le("self_flat_ripple_db", float(np.max(db) - np.min(db)), 0.01))
    tilt_power = (freq / 1000.0) ** (3.0 / 10.0 * math.log(10) / math.log(2))
    # power ~ f^(slope/10/log2(e)*something). Easier: build dB then convert.
    # 3 dB/oct means power ratio = 10^((3*log2(f/fref))/10) = (f/fref)^(3/10/log10(2))? 
    # 10*log10(p2/p1) = 3 * log2(f2/f1) = 3 * ln(f2/f1)/ln(2)
    # log10(p2/p1) = 0.3 * log2(f2/f1)
    # p ~ f ^ (0.3 * log(2)? wait)
    # log2(p) = ln(p)/ln(2). Let p = f^a. 10*log10(p) = 10*a*log10(f) = 3*log2(f) = 3*log10(f)/log10(2)
    # 10*a = 3/log10(2) => a = 0.3 / log10(2) ≈ 0.9966. Approximately 1.
    a = 3.0 / (10.0 * math.log10(2.0))
    tilt_power = (freq / 1000.0) ** a
    c, sm = smooth_mean(freq, tilt_power, 3.0, 200.0, 10000.0)
    db = 10.0 * np.log10(np.maximum(sm, 1e-30))
    ok = np.isfinite(db)
    got = float(np.polyfit(np.log2(c[ok]), db[ok], 1)[0])
    metrics.append(m_abs_le("self_tilt_error_db_per_oct", got - 3.0, 0.15))

    # Correlator sign and scale: shift a two-bump envelope up by exactly 7 st.
    s = np.arange(0.0, 47.0, 0.25)
    env = np.exp(-0.5 * ((s - 12.0) / 2.0) ** 2) + 0.6 * np.exp(-0.5 * ((s - 28.0) / 3.0) ** 2)
    env_db = 20.0 * np.log10(env + 1e-3)
    moved = np.interp(s - 7.0, s, env_db, left=np.nan, right=np.nan)
    shift, _corr = measure_shift_st(env_db, moved, 0.25)
    metrics.append(m_abs_le("self_shift_error_st", shift - 7.0, 0.2))

    # Same estimator on a spectrally shifted noise burst.
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(int(3.0 * sr))
    shaped = signals._formant(noise, sr, 800.0, 180.0)
    shaped = signals._formant(shaped, sr, 1800.0, 220.0)
    shifted = shift_spectrum(shaped, 7.0)
    _sa, ea = envelope_db(shaped, sr)
    _sb, eb = envelope_db(shifted, sr)
    ashift, acorr = measure_shift_st(ea, eb, 0.25)
    metrics.append(m_abs_le("self_audio_shift_error_st", ashift - 7.0, 1.0))
    metrics.append(m_info("self_audio_shift_corr", acorr))

    # YIN on a pure sine and on the synthetic voice (the input, not the plugin).
    sine = signals.sine(int(2.0 * sr), sr, 220.0, -6.0)
    _t, est, _lv = track_f0(sine, sr, fmin=50.0, fmax=1000.0, gate_db=-40.0)
    err = 1200.0 * np.log2(est[np.isfinite(est)] / 220.0)
    med = float(np.median(np.abs(err))) if len(err) else float("inf")
    metrics.append(m_le("self_yin_sine_cents", med, 2.0))

    voice = ctx.cat["voice"]
    times, est, _lv = track_f0(voice["audio"], sr, fmin=50.0, fmax=1000.0, gate_db=-40.0)
    frame_len = int(round(2.0 * sr / 50.0))
    vmed, oct_pct, n_voiced = _pitch_report(est, voice["f0"], voice["voiced"], times, sr, frame_len)
    metrics.append(m_info("self_yin_voiced_frames", n_voiced))
    metrics.append(m_le("self_yin_voice_median_cents", vmed, 20.0))
    metrics.append(m_le("self_yin_voice_octave_pct", oct_pct, 1.0))

    # Known-flat carrier passthrough: ripple must collapse.
    car = ctx.cat["white_carrier_10s"]
    mod = ctx.cat["white_mod_10s"]
    audio = signals.assemble(mod, mod, car, car)
    # Render the fixture directly, bypassing the vocoder slider map.
    saved_plugin, saved_v1 = ctx.plugin, ctx.v1
    ctx.plugin = ROOT / "tests" / "fixtures" / "passthrough_carrier.jsfx"
    ctx.v1 = False
    try:
        y, meta = render(ctx, audio, {}, "v0_pass")
    finally:
        ctx.plugin, ctx.v1 = saved_plugin, saved_v1
    rip3, med = ripple_of_ratio(y, car, sr, 100.0, 14400.0, 3.0)
    rip12, _m = ripple_of_ratio(y, car, sr, 100.0, 14400.0, 12.0)
    metrics.append(m_le("self_passthrough_ripple_1_3_db", rip3, 0.2))
    metrics.append(m_le("self_passthrough_ripple_1_12_db", rip12, 0.5))
    metrics.append(m_abs_le("self_passthrough_ratio_median_db", med, 0.2))

    ctx.plugin = ROOT / "tests" / "fixtures" / "zero_output.jsfx"
    ctx.v1 = False
    try:
        z, _meta = render(ctx, np.zeros((int(0.5 * sr), 4)), {}, "v0_zero")
    finally:
        ctx.plugin, ctx.v1 = saved_plugin, saved_v1
    metrics.append(m_le("self_zero_peak", float(np.max(np.abs(z))), 0.0))
    return metrics


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("T1", "Filterbank flatness", test_t1),
    ("T2", "Envelope ripple", test_t2),
    ("T3", "Enhance tilt", test_t3),
    ("T4", "Resynthesis", test_t4),
    ("T5", "Level vs bands", test_t5),
    ("T6", "Gate", test_t6),
    ("T7", "Unvoiced", test_t7),
    ("T8", "Pitch tracking", test_t8),
    ("T9", "Channel modes", test_t9),
    ("T10", "Robustness", test_t10),
    ("T11a", "CPU default", test_t11a),
    ("T11b", "CPU worst case", test_t11b),
    ("T12", "Formant", test_t12),
    ("T13", "Noise carrier", test_t13),
    ("T14", "Level sanity", test_t14),
    ("T15", "Release and silence", test_t15),
    ("T16", "Attack", test_t16),
]


def status_of(metrics: list) -> str:
    if not metrics:
        return "FAIL"
    if any(m["status"] == "FAIL" for m in metrics):
        return "FAIL"
    if all(m["status"] == "NA" for m in metrics):
        return "N/A"
    if any(m["status"] == "PASS" for m in metrics):
        return "PASS"
    return "INFO"


def fmt_value(v):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    av = abs(float(v))
    if av != 0 and (av < 1e-3 or av >= 1e4):
        return f"{float(v):.3e}"
    return f"{float(v):.4f}"


def print_table(rows):
    headers = ("test", "metric", "value", "threshold", "result")
    body = []
    for test, m in rows:
        body.append((test, m["name"], fmt_value(m["value"]), m["threshold"], m["status"]))
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    print(line(headers))
    print(line(tuple("-" * w for w in widths)))
    for row in body:
        print(line(row))


def ysfx_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT / "tools" / "ysfx"), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class Ctx:
    def __init__(self, plugin, v1, sr, block, render_bin, cat):
        self.plugin = Path(plugin)
        self.v1 = v1
        self.sr = sr
        self.block = block
        self.render_bin = Path(render_bin)
        self.cat = cat


def run_one(tid, title, fn, ctx):
    t0 = time.perf_counter()
    try:
        metrics = fn(ctx)
        status = status_of(metrics)
        err = ""
    except MissingFeature as exc:
        metrics = [metric("feature", None, "n/a", "NA", str(exc))]
        status = "N/A"
        err = str(exc)
    except (RenderError, subprocess.TimeoutExpired, OSError) as exc:
        metrics = [metric("error", None, "render", "FAIL", str(exc).splitlines()[0][:240])]
        status = "FAIL"
        err = str(exc)
    dt = time.perf_counter() - t0
    print(f"[{status}] {tid} {title} ({dt:.1f}s)", file=sys.stderr, flush=True)
    return {
        "id": tid,
        "title": title,
        "status": status,
        "seconds": round(dt, 3),
        "error": err,
        "metrics": metrics,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vocoder offline suite")
    ap.add_argument("--plugin", default=None, help="JSFX path (default depends on --v1)")
    ap.add_argument("--only", default="", help="Comma-separated ids, e.g. T1,T3")
    ap.add_argument("--v1", action="store_true", help="Run AbletonVocoder_v1.jsfx with the v1 slider map")
    args = ap.parse_args(argv)

    if not RENDER.is_file():
        print(f"renderer missing: {RENDER}\nRun tools/build.sh first.", file=sys.stderr)
        return 1
    plugin = Path(args.plugin) if args.plugin else (ROOT / ("AbletonVocoder_v1.jsfx" if args.v1 else "AbletonVocoder.jsfx"))
    if not plugin.is_absolute():
        plugin = ROOT / plugin
    if not plugin.is_file():
        print(f"plugin missing: {plugin}", file=sys.stderr)
        return 1

    only = {p.strip().upper() for p in args.only.split(",") if p.strip()}
    print("generating signals...", file=sys.stderr, flush=True)
    cat = signals.generate_catalog(48000)
    ctx = Ctx(plugin, args.v1, 48000, BLOCK, RENDER, cat)

    print("validating metrics...", file=sys.stderr, flush=True)
    vrec = run_one("V0", "Metric self-check", validate_clean, ctx)
    results = [vrec]
    if vrec["status"] == "FAIL":
        _finish(results, ctx, aborted=True)
        return 1

    for tid, title, fn in TESTS:
        if only and tid.upper() not in only:
            continue
        results.append(run_one(tid, title, fn, ctx))

    _finish(results, ctx, aborted=False)
    failed = [r for r in results if r["status"] == "FAIL"]
    return 2 if failed else 0


def _finish(results, ctx, aborted: bool):
    rows = []
    for rec in results:
        for m in rec["metrics"]:
            rows.append((rec["id"], m))
    print_table(rows)
    counts = {}
    for rec in results:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"\n{summary}" + ("  aborted after failed self-check" if aborted else ""))
    try:
        plugin_s = str(ctx.plugin.resolve().relative_to(ROOT))
    except ValueError:
        plugin_s = str(ctx.plugin)
    report = {
        "plugin": plugin_s,
        "v1": ctx.v1,
        "sample_rate": ctx.sr,
        "block": ctx.block,
        "ysfx_commit": ysfx_commit(),
        "summary": counts,
        "aborted": aborted,
        "tests": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
