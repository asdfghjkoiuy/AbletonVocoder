"""Route changes must not fall back to the dry signal or click the right channel.

Switches Channels 0 -> 1 -> 2 -> 1 (the last step is the fractional value
0.999, which has to quantize to Stereo). A second render holds Channels at
1.2 for the whole file. A third render is Noise + Crackle in Stereo, where a
per-sample carrier comparison used to retune the right filter every impulse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signals
import suite

SR = 48000
DUR = 2.0


def _rms(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def _rms_db(x):
    r = _rms(x)
    if r <= 1e-20:
        return -240.0
    return 20.0 * np.log10(r)


def _window_rms(x, sr, win_s, hop_s):
    x = np.asarray(x, dtype=np.float64)
    w = max(1, int(round(win_s * sr)))
    h = max(1, int(round(hop_s * sr)))
    if len(x) < w:
        return np.array([_rms(x)])
    return np.array([_rms(x[i:i + w]) for i in range(0, len(x) - w + 1, h)])


def _rel_dry(out, dry):
    d = _rms(dry)
    if d < 1e-8:
        return 1.0
    return _rms(out - dry) / d


def _d2(x):
    return np.diff(np.asarray(x, dtype=np.float64), n=2)


def main() -> int:
    ctx = suite.Ctx(
        suite.ROOT / "AbletonVocoder.jsfx",
        False,
        SR,
        suite.BLOCK,
        suite.RENDER,
        {},
    )
    n = int(DUR * SR)
    rng = np.random.default_rng(3)
    mod = signals.pink(n, rng, rms_db=-18.0)
    car_l = signals.white(n, np.random.default_rng(4), rms_db=-18.0)
    car_r = signals.white(n, np.random.default_rng(5), rms_db=-18.0)
    audio = signals.assemble(mod, mod, car_l, car_r)
    csv = suite.OUT_DIR / "route_switch.csv"
    suite.OUT_DIR.mkdir(parents=True, exist_ok=True)
    # slider 14 = Channels. 0.999 must quantize to Stereo (1), not stick.
    lines = [
        "sample_offset,slider_idx,value",
        f"{int(0.50 * SR)},14,1",
        f"{int(1.00 * SR)},14,2",
        f"{int(1.50 * SR)},14,0.999",
    ]
    csv.write_text("\n".join(lines) + "\n")
    y, meta = suite.render(ctx, audio, {
        "carrier": "external",
        "channels": "mono",
        "enhance": 0,
        "unvoiced": -96,
        "drywet": 100,
        "depth": 100,
    }, "route_switch", sr=SR, automate=csv)
    fails = []
    if int(meta.get("nan_count", 0)) or int(meta.get("inf_count", 0)):
        fails.append("nan/inf in route switch")

    dry = np.stack([mod, mod], axis=1)
    spans = {
        "mono": (0.15, 0.42),
        "stereo": (0.62, 0.92),
        "lr": (1.12, 1.42),
        "stereo_frac": (1.62, 1.92),
    }
    steady_rms = []
    for name, (a, b) in spans.items():
        sl = slice(int(a * SR), int(b * SR))
        seg = y[sl]
        level = _rms_db(seg)
        steady_rms.append(_rms(seg))
        err = _rel_dry(seg, dry[sl])
        corr = float(np.corrcoef(seg[:, 0], dry[sl, 0])[0, 1])
        print(f"{name}: rms {level:.2f} dBFS  dry_rel {err:.3f}  corr {corr:.3f}")
        if level < -40.0:
            fails.append(f"{name} is too quiet ({level:.1f} dBFS), route may be stuck ducked")
        if err < 0.5:
            fails.append(f"{name} matches the dry modulator (rel {err:.3f})")
        if abs(corr) > 0.8:
            fails.append(f"{name} correlates with dry ({corr:.3f})")
        if name in ("stereo", "lr", "stereo_frac"):
            r_db = _rms_db(seg[:, 1])
            print(f"  right {r_db:.2f} dBFS")
            if r_db < -40.0:
                fails.append(f"{name} right channel is silent ({r_db:.1f} dBFS)")
        if name == "mono":
            diff = float(np.max(np.abs(seg[:, 0] - seg[:, 1])))
            print(f"  mono L-R max {diff:.3e}")
            if diff > 1e-5:
                fails.append(f"mono channels differ by {diff:.3e}")

    med = float(np.median(steady_rms))
    switches = (0.50, 1.00, 1.50)
    d2 = _d2(y[:, 1])
    steady_abs = []
    for a, b in spans.values():
        s0 = int(a * SR)
        s1 = int(b * SR)
        steady_abs.append(np.abs(d2[s0:s1 - 2]))
    steady_abs = np.concatenate(steady_abs)
    p99 = float(np.quantile(steady_abs, 0.995))
    print(f"steady right |d2| p99.5 {p99:.5f}  median steady rms {med:.5f}")
    for t in switches:
        a = int((t - 0.04) * SR)
        b = int((t + 0.05) * SR)
        hop = _window_rms(y[a:b], SR, 0.002, 0.001)
        dip = float(np.min(hop)) if len(hop) else 0.0
        dip_db = -240.0 if dip <= 1e-20 else 20.0 * np.log10(dip)
        med_db = 20.0 * np.log10(med + 1e-20)
        print(f"switch {t:.2f}s: min 2ms rms {dip_db:.2f} dBFS (steady {med_db:.2f})")
        if dip > med * 10 ** (-12.0 / 20.0):
            fails.append(
                f"switch at {t:.2f}s only dipped to {dip_db:.1f} dBFS "
                f"(steady {med_db:.1f}); duck may be crossfading to dry"
            )
        sl = slice(max(0, a), min(len(d2), b))
        peak = float(np.max(np.abs(d2[sl]))) if sl.stop > sl.start else 0.0
        limit = max(0.05, p99 * 12.0)
        print(f"  right |d2| peak {peak:.5f}  limit {limit:.5f}")
        if peak > limit:
            fails.append(f"right-channel click at {t:.2f}s (|d2| {peak:.4f} > {limit:.4f})")
        if peak > 1.0:
            fails.append(f"right-channel blast at {t:.2f}s (|d2| {peak:.3f})")

    # Fractional Channels held at 1.2 must behave as Stereo, not permanent dry.
    y_frac, _meta = suite.render(ctx, audio[: SR], {
        "carrier": "external",
        "channels": "stereo",
        "enhance": 0,
        "unvoiced": -96,
    }, "route_frac_base", sr=SR)
    cmd_audio = audio[:SR]
    # Overwrite slider 14 after map_sliders by a direct render with 1.2.
    in_path = suite.OUT_DIR / "route_frac_in.wav"
    out_path = suite.OUT_DIR / "route_frac_out.wav"
    signals.write_wav(in_path, cmd_audio, SR)
    sets = suite.map_sliders({
        "carrier": "external",
        "channels": "stereo",
        "enhance": 0,
        "unvoiced": -96,
    }, False)
    cmd = [str(ctx.render_bin), str(ctx.plugin), str(in_path), str(out_path), "--sr", str(SR), "--block", str(suite.BLOCK)]
    for idx, val in sets:
        if idx == 14:
            val = 1.2
        cmd.extend(["--set", f"{idx}={val:.10g}"])
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        fails.append("fractional channels render failed")
    else:
        y12, _ysr = __import__("soundfile").read(str(out_path), always_2d=True, dtype="float64")
        sl = slice(int(0.25 * SR), None)
        err = _rel_dry(y12[sl], dry[:SR][sl])
        level = _rms_db(y12[sl])
        # 1.2 quantizes to the same Stereo path as an integer 1.
        ref = y_frac[sl]
        n = min(len(y12[sl]), len(ref))
        delta = _rms(y12[sl][:n] - ref[:n])
        ref_r = _rms(ref[:n]) + 1e-12
        print(f"channels 1.2: rms {level:.2f} dBFS  dry_rel {err:.3f}  vs stereo rel {delta / ref_r:.4f}")
        if err < 0.5 or level < -40.0:
            fails.append("channels=1.2 fell back to dry or silence")
        if delta / ref_r > 0.02:
            fails.append(f"channels=1.2 diverges from Stereo (rel {delta / ref_r:.3f})")

    # Crackle must not blast the right channel. Both sides are impulses.
    rng = np.random.default_rng(7)
    mod_n = signals.pink(SR, rng, rms_db=-18.0)
    noise = signals.assemble(mod_n, mod_n, np.zeros(SR), np.zeros(SR))
    y_c, _mc = suite.render(ctx, noise, {
        "carrier": "noise",
        "channels": "stereo",
        "noise_crackle": 0.25,
        "noise_rate": 1.0,
        "enhance": 0,
        "unvoiced": -96,
    }, "route_crackle", sr=SR)
    sl = slice(int(0.1 * SR), None)
    peak_l = float(np.max(np.abs(y_c[sl, 0])))
    peak_r = float(np.max(np.abs(y_c[sl, 1])))
    d2_l = float(np.max(np.abs(_d2(y_c[sl, 0]))))
    d2_r = float(np.max(np.abs(_d2(y_c[sl, 1]))))
    print(f"crackle peak L {peak_l:.4f} R {peak_r:.4f}  |d2| L {d2_l:.4f} R {d2_r:.4f}")
    if peak_r > max(1.0, peak_l * 8.0):
        fails.append(f"crackle right peak {peak_r:.3f} vs left {peak_l:.3f}")
    if d2_r > max(1.0, d2_l * 8.0):
        fails.append(f"crackle right click {d2_r:.3f} vs left {d2_l:.3f}")

    if fails:
        print("FAIL")
        for item in fails:
            print(" -", item)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
