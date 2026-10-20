"""Deterministic test signals for the vocoder harness.

Generated wavs go to tests/signals/ (gitignored). Run:

    tools/.venv/bin/python tools/signals.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SIGNAL_DIR = ROOT / "tests" / "signals"

SPEECH_TEXT = (
    "Sixty six sisters sing softly. "
    "The quick brown fox jumps over the lazy dog. "
    "She sells sea shells."
)

# Kellet pink-noise IIR (musicdsp). White in, about -3 dB/oct out.
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])


def lfilter(b, a, x) -> np.ndarray:
    """Direct-form-II-transposed IIR. scipy.signal.lfilter is not imported:
    the scipy 1.15 macOS arm64 wheel does not load on this dyld."""
    b = np.asarray(b, dtype=np.float64).copy()
    a = np.asarray(a, dtype=np.float64).copy()
    x = np.asarray(x, dtype=np.float64)
    if a[0] != 1.0:
        b /= a[0]
        a /= a[0]
    n_z = max(len(b), len(a)) - 1
    if n_z <= 0:
        return (b[0] * x) if len(b) else np.zeros_like(x)
    if len(b) < n_z + 1:
        b = np.pad(b, (0, n_z + 1 - len(b)))
    if len(a) < n_z + 1:
        a = np.pad(a, (0, n_z + 1 - len(a)))
    y = np.empty(len(x))
    z = np.zeros(n_z)
    for i, xi in enumerate(x):
        yi = b[0] * xi + z[0]
        y[i] = yi
        for k in range(n_z - 1):
            z[k] = b[k + 1] * xi - a[k + 1] * yi + z[k + 1]
        z[n_z - 1] = b[n_z] * xi - a[n_z] * yi
    return y


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    sf.write(str(path), data, sr, subtype="FLOAT")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False, dtype="float64")
    return np.asarray(audio, dtype=np.float64), int(sr)


def _scale_rms(x: np.ndarray, rms_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x * x)) + 1e-20)
    return x * (db_to_amp(rms_db) / rms)


def _scale_peak(x: np.ndarray, peak: float) -> np.ndarray:
    p = float(np.max(np.abs(x)) + 1e-20)
    return x * (peak / p)


def white(n: int, rng: np.random.Generator, rms_db: float = -12.0) -> np.ndarray:
    return _scale_rms(rng.standard_normal(n), rms_db)


def pink(n: int, rng: np.random.Generator, rms_db: float = -12.0) -> np.ndarray:
    x = rng.standard_normal(n)
    y = lfilter(_PINK_B, _PINK_A, x)
    return _scale_rms(y, rms_db)


def sine(n: int, sr: int, freq: float, peak_db: float = -12.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    return db_to_amp(peak_db) * np.sin(2.0 * np.pi * freq * t)


def square(n: int, sr: int, freq: float, amp: float = 1.0) -> np.ndarray:
    phase = np.mod(np.arange(n, dtype=np.float64) * (freq / sr), 1.0)
    return np.where(phase < 0.5, amp, -amp).astype(np.float64)


def bandlimited_saw(n: int, sr: int, freq: float, peak: float = 0.5) -> np.ndarray:
    """Fourier saw, harmonics up to Nyquist. Falling slope, no DC."""
    nh = max(1, int((sr * 0.5) / freq) - 1)
    phase = 2.0 * np.pi * freq * np.arange(n, dtype=np.float64) / sr
    z = np.exp(1j * phase)
    w = z.copy()
    acc = np.zeros(n, dtype=np.float64)
    for k in range(1, nh + 1):
        acc += w.imag / k
        w *= z
    return _scale_peak((-2.0 / np.pi) * acc, peak)


def saw_minor_chord(n: int, sr: int, root: float = 110.0, peak: float = 0.5) -> np.ndarray:
    freqs = [root, root * 2.0 ** (3.0 / 12.0), root * 2.0 ** (7.0 / 12.0)]
    acc = np.zeros(n, dtype=np.float64)
    for f in freqs:
        acc += bandlimited_saw(n, sr, f, peak=1.0)
    return _scale_peak(acc, peak)


def _rosenberg(sr: int, f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    oq = 0.6
    phase = 0.0
    prev = 0.0
    y = np.zeros(len(f0), dtype=np.float64)
    for i in range(len(f0)):
        if not voiced[i] or f0[i] <= 1.0:
            phase = 0.0
            prev = 0.0
            continue
        phase += float(f0[i]) / sr
        if phase >= 1.0:
            phase -= 1.0
        if phase < oq:
            g = 0.5 * (1.0 - np.cos(np.pi * phase / oq))
        else:
            g = np.cos(0.5 * np.pi * (phase - oq) / (1.0 - oq))
        y[i] = g - prev
        prev = g
    return y


def _formant(x: np.ndarray, sr: int, freq: float, bw: float) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    b = np.array([1.0 - r])
    a = np.array([1.0, -2.0 * r * np.cos(2.0 * np.pi * freq / sr), r * r])
    return lfilter(b, a, x)


def _piece(sr: int, dur: float, f_start: float, f_end: float, vib_hz: float, vib_cents: float, voiced: bool):
    n = int(round(dur * sr))
    t = np.arange(n, dtype=np.float64) / sr
    if not voiced or f_start <= 0:
        return np.zeros(n), np.zeros(n, dtype=bool)
    if abs(f_end - f_start) < 1e-9:
        base = np.full(n, f_start)
    else:
        base = f_start * (f_end / f_start) ** (t / max(t[-1], 1e-12))
    vib = 2.0 ** ((vib_cents / 1200.0) * np.sin(2.0 * np.pi * vib_hz * t))
    return base * vib, np.ones(n, dtype=bool)


def synth_voice(sr: int) -> dict:
    """Glottal pulse train, f0 glides 100→300 Hz with vibrato, three formants, silent gaps.

    Returns audio, per-sample f0 (0 in gaps), voiced mask, and gap sample ranges.
    """
    parts = [
        _piece(sr, 0.40, 0, 0, 0, 0, False),
        _piece(sr, 2.00, 100.0, 300.0, 5.5, 15.0, True),
        _piece(sr, 0.50, 0, 0, 0, 0, False),
        _piece(sr, 1.20, 180.0, 180.0, 5.5, 15.0, True),
        _piece(sr, 0.40, 0, 0, 0, 0, False),
    ]
    f0 = np.concatenate([p[0] for p in parts])
    voiced = np.concatenate([p[1] for p in parts])
    excit = _rosenberg(sr, f0, voiced)
    y = excit
    for freq, bw in ((700.0, 80.0), (1200.0, 100.0), (2600.0, 160.0)):
        y = _formant(y, sr, freq, bw)
    y = np.where(voiced, y, 0.0)
    y = _scale_peak(y, 0.4)

    gaps = []
    in_gap = False
    start = 0
    for i, v in enumerate(voiced):
        if not v and not in_gap:
            in_gap = True
            start = i
        elif v and in_gap:
            gaps.append((start, i))
            in_gap = False
    if in_gap:
        gaps.append((start, len(voiced)))
    return {"audio": y, "f0": f0, "voiced": voiced, "gaps": gaps, "sr": sr}


def decorr_stereo(n: int, seed_l: int, seed_r: int, rms_db: float = -12.0) -> tuple[np.ndarray, np.ndarray]:
    left = white(n, np.random.default_rng(seed_l), rms_db)
    right = white(n, np.random.default_rng(seed_r), rms_db)
    return left, right


def assemble(mod_l, mod_r, car_l, car_r) -> np.ndarray:
    cols = [np.asarray(c, dtype=np.float64) for c in (mod_l, mod_r, car_l, car_r)]
    n = min(len(c) for c in cols)
    return np.stack([c[:n] for c in cols], axis=1)


def _trim_silence(x: np.ndarray, sr: int, thresh_db: float = -45.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    frame = max(1, int(0.02 * sr))
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-20)
    db = 20.0 * np.log10(rms)
    hot = np.where(db > thresh_db)[0]
    if len(hot) == 0:
        return x
    a = max(0, hot[0] * frame - frame)
    b = min(len(x), (hot[-1] + 1) * frame + frame)
    return x[a:b]


def ensure_speech(sr: int) -> np.ndarray:
    """macOS say → ffmpeg. Cached per sample rate under tests/signals/."""
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = SIGNAL_DIR / f"speech_{sr}.wav"
    meta_path = SIGNAL_DIR / "speech_source.json"
    fresh = dest.exists() and meta_path.exists()
    if fresh:
        try:
            meta = json.loads(meta_path.read_text())
            fresh = meta.get("text") == SPEECH_TEXT and sr in meta.get("rates", [])
        except json.JSONDecodeError:
            fresh = False
    if not fresh:
        aiff = SIGNAL_DIR / "speech.aiff"
        subprocess.run(
            ["say", "-v", "Samantha", "-o", str(aiff), SPEECH_TEXT],
            check=True,
        )
        rates = []
        if meta_path.exists():
            try:
                rates = json.loads(meta_path.read_text()).get("rates", [])
            except json.JSONDecodeError:
                rates = []
        for rate in (44100, 48000, 96000):
            out = SIGNAL_DIR / f"speech_{rate}.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(aiff),
                    "-ac", "1", "-ar", str(rate),
                    "-c:a", "pcm_f32le",
                    str(out),
                ],
                check=True,
            )
            if rate not in rates:
                rates.append(rate)
        meta_path.write_text(json.dumps({"text": SPEECH_TEXT, "rates": rates, "voice": "Samantha"}, indent=2))
    audio, file_sr = read_wav(dest)
    if file_sr != sr:
        raise RuntimeError(f"{dest} is {file_sr} Hz, wanted {sr}")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return _trim_silence(audio, sr)


def speech_with_noise_floor(sr: int, seed: int = 1006) -> dict:
    """Speech with known gaps filled by a -50 dBFS noise floor.

    Measurement windows skip the release tail: pauses are scored on the last
    portion of each 1.2 s gap, speech windows skip the first 80 ms.
    """
    speech = ensure_speech(sr)
    if len(speech) < int(1.0 * sr):
        raise RuntimeError("speech is unexpectedly short")
    mid = len(speech) // 2
    # Keep each half long enough to contain vowels and fricatives, but not huge.
    half = min(mid, int(2.5 * sr))
    a = speech[:half]
    b = speech[mid : mid + half]
    gap_n = int(1.20 * sr)
    pad_n = int(1.00 * sr)
    rng = np.random.default_rng(seed)
    noise_rms = db_to_amp(-50.0)

    def noise(n: int) -> np.ndarray:
        return _scale_rms(rng.standard_normal(n), -50.0) if n else np.zeros(0)

    # scale check: _scale_rms to -50 dBFS RMS. `noise_rms` documents the target.
    del noise_rms
    parts = [noise(pad_n), a, noise(gap_n), b, noise(pad_n)]
    y = np.concatenate(parts)
    # Index map.
    lengths = [len(p) for p in parts]
    edges = np.cumsum([0] + lengths)
    # parts: 0 pad, 1 speech A, 2 gap, 3 speech B, 4 pad
    release_skip = int(0.70 * sr)
    speech_skip = int(0.08 * sr)
    pauses = []
    for idx in (0, 2, 4):
        s, e = int(edges[idx]), int(edges[idx + 1])
        if e - s <= release_skip + int(0.15 * sr):
            continue
        pauses.append((s + release_skip, e - int(0.02 * sr)))
    speech_regions = []
    for idx in (1, 3):
        s, e = int(edges[idx]), int(edges[idx + 1])
        speech_regions.append((s + speech_skip, e - int(0.02 * sr)))
    return {"audio": y, "pauses": pauses, "speech": speech_regions, "sr": sr}


def save_f0_csv(path: Path, f0: np.ndarray, voiced: np.ndarray, sr: int, hop_s: float = 0.01) -> None:
    hop = max(1, int(round(hop_s * sr)))
    lines = ["time_s,f0_hz,voiced"]
    for i in range(0, len(f0), hop):
        lines.append(f"{i / sr:.6f},{f0[i]:.6f},{1 if voiced[i] else 0}")
    path.write_text("\n".join(lines) + "\n")


def generate_catalog(sr: int = 48000) -> dict:
    """Write the standard signal set and return the in-memory arrays the suite uses."""
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    rng_c = np.random.default_rng(1001)
    rng_m = np.random.default_rng(1002)
    n10 = int(10.0 * sr)
    n4 = int(4.0 * sr)
    carrier = white(n10, rng_c, rms_db=-12.0)
    modulator = white(n10, rng_m, rms_db=-12.0)
    write_wav(SIGNAL_DIR / f"white_carrier_{sr}.wav", carrier, sr)
    write_wav(SIGNAL_DIR / f"white_mod_{sr}.wav", modulator, sr)
    write_wav(SIGNAL_DIR / f"pink_{sr}.wav", pink(n4, np.random.default_rng(1003), -12.0), sr)

    saw = bandlimited_saw(n4, sr, 110.0, peak=0.5)
    chord = saw_minor_chord(n4, sr, 110.0, peak=0.5)
    write_wav(SIGNAL_DIR / f"saw110_{sr}.wav", saw, sr)
    write_wav(SIGNAL_DIR / f"saw_minor_{sr}.wav", chord, sr)

    n2 = int(2.0 * sr)
    s250 = sine(n2, sr, 250.0, peak_db=-12.0)
    s262 = sine(n2, sr, 262.0, peak_db=-12.0)
    write_wav(SIGNAL_DIR / f"sine250_{sr}.wav", s250, sr)
    write_wav(SIGNAL_DIR / f"sine262_{sr}.wav", s262, sr)

    voice = synth_voice(sr)
    write_wav(SIGNAL_DIR / f"synth_voice_{sr}.wav", voice["audio"], sr)
    save_f0_csv(SIGNAL_DIR / f"synth_voice_f0_{sr}.csv", voice["f0"], voice["voiced"], sr)

    speech = ensure_speech(sr)
    write_wav(SIGNAL_DIR / f"speech_trimmed_{sr}.wav", speech, sr)
    # Also materialise the other two speech rates so the catalog matches the spec,
    # even when the suite itself renders at `sr`.
    for other in (44100, 48000, 96000):
        if other != sr:
            ensure_speech(other)

    floor = speech_with_noise_floor(sr)
    write_wav(SIGNAL_DIR / f"speech_floor_{sr}.wav", floor["audio"], sr)
    (SIGNAL_DIR / f"speech_floor_{sr}.json").write_text(json.dumps({
        "sr": sr,
        "pauses": floor["pauses"],
        "speech": floor["speech"],
        "noise_floor_dbfs": -50,
    }, indent=2))

    dl, dr = decorr_stereo(n4, 1004, 1005, rms_db=-12.0)
    write_wav(SIGNAL_DIR / f"decorr_{sr}.wav", np.stack([dl, dr], axis=1), sr)

    return {
        "sr": sr,
        "white_carrier_10s": carrier,
        "white_mod_10s": modulator,
        "saw": saw,
        "chord": chord,
        "sine250": s250,
        "sine262": s262,
        "voice": voice,
        "speech": speech,
        "floor": floor,
    }


if __name__ == "__main__":
    catalog = generate_catalog(48000)
    print(f"wrote signals to {SIGNAL_DIR}")
    print(f"speech samples: {len(catalog['speech'])}")
    print(f"synth samples: {len(catalog['voice']['audio'])} gaps: {catalog['voice']['gaps']}")
