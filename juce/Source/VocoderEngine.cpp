#include "VocoderEngine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kLn2 = 0.6931471805599453;
constexpr double kDb2A = 2.302585092994046 / 20.0;
constexpr double kQDen = 1.553773974030037307; // sqrt(1 + sqrt(2))
constexpr float kGainCap = 32.f;
constexpr float kRefLin = 0.12589254117941673f;
constexpr float kInvRefDens = 1.f / kRefLin;
constexpr float kNominal = 0.05f;
constexpr float kInvRefEnh = 1.f / kNominal;
constexpr float kEnhRel = 0.001f;
constexpr float kEnhAbs = 1.0e-5f;
constexpr float kEnhKnee = 3.9810717055349722f;
constexpr float kTrim[4] = {
    0.6831955045222029f,
    0.11665812717409325f,
    0.05570895346700124f,
    0.07670081876935099f
};
constexpr float kNoiseAmp = 0.21805227761670687f;
constexpr float kWhiteRms = 0.12589254117941673f;
constexpr float kSaw = 0.21805227761670687f;
constexpr float kP50 = 0.12589254117941673f;
constexpr float kP25 = 0.14536818507780461f;
constexpr float kP125 = 0.19033163193068273f;

struct SvfC
{
    double a1, a2, a3, k;
};

SvfC svfSet(double fc, double qq, double sr)
{
    fc = std::clamp(fc, 1.0, sr * 0.45);
    qq = std::max(qq, 0.12);
    const double g = std::tan(kPi * fc / sr);
    const double kk = 1.0 / qq;
    const double a1 = 1.0 / (1.0 + g * (g + kk));
    const double a2 = g * a1;
    const double a3 = g * a2;
    return { a1, a2, a3, kk };
}

float onePole(double timeSec, double sr)
{
    return (float) (1.0 - std::exp(-1.0 / (timeSec * sr)));
}

// Non-finite input falls back. Finite values use std::clamp, including values
// outside [lo, hi]. The finite test is a bitmask so -ffast-math cannot turn
// NaN into a "successful" comparison that poisons the smoother forever.
float safeClamp(float x, float lo, float hi, float fallback) noexcept
{
    if (!vocIsFinite(x))
    {
        if (!vocIsFinite(fallback))
            return std::clamp(0.f, lo, hi);
        return std::clamp(fallback, lo, hi);
    }
    return std::clamp(x, lo, hi);
}

void healScalar(float& x, float fallback) noexcept
{
    if (!vocIsFinite(x))
        x = vocIsFinite(fallback) ? fallback : 0.f;
}

#if defined(__aarch64__)
struct NoDenorms
{
    uint64_t old;
    NoDenorms() noexcept
    {
        uint64_t fpcr = 0;
        asm volatile("mrs %0, fpcr" : "=r"(fpcr));
        old = fpcr;
        fpcr |= (1ull << 24);
        asm volatile("msr fpcr, %0" :: "r"(fpcr));
    }
    ~NoDenorms() noexcept
    {
        asm volatile("msr fpcr, %0" :: "r"(old));
    }
};
#elif VOC_SSE2
struct NoDenorms
{
    unsigned int old;
    NoDenorms() noexcept : old(_mm_getcsr())
    {
        _mm_setcsr(old | 0x8040u); // FTZ + DAZ
    }
    ~NoDenorms() noexcept
    {
        _mm_setcsr(old);
    }
};
#else
struct NoDenorms
{
    NoDenorms() noexcept {}
    ~NoDenorms() noexcept {}
};
#endif

VOC_INLINE V4 svf2(V4 x, V4 a1, V4 a2, V4 a3, V4 k,
                   V4& ic1, V4& ic2, V4& jc1, V4& jc2, V4& power) noexcept
{
    const V4 two = V4::splat(2.f);
    V4 v3 = x - ic2;
    V4 v1 = a1 * ic1 + a2 * v3;
    V4 v2 = ic2 + a2 * ic1 + a3 * v3;
    ic1 = two * v1 - ic1;
    ic2 = two * v2 - ic2;
    V4 bp = k * v1;

    v3 = bp - jc2;
    v1 = a1 * jc1 + a2 * v3;
    v2 = jc2 + a2 * jc1 + a3 * v3;
    jc1 = two * v1 - jc1;
    jc2 = two * v2 - jc2;
    bp = k * v1;
    const V4 lp = k * v2;
    power = bp * bp + lp * lp;
    return bp;
}

VOC_INLINE V4 followEnv(V4 env, V4 power, V4 attC, V4 relC) noexcept
{
    const V4 zero = V4::splat(0.f);
    const V4 amp = vsqrt(vmax(power * V4::splat(0.5f), zero));
    const V4 c = vsel(vgt(amp, env), attC, relC);
    env = env + c * (amp - env);
    env = vmax(env, zero);
    return vsel(vlt(env, V4::splat(1.0e-12f)), zero, env);
}

VOC_INLINE void followGate(V4& open, V4& gate, V4 env, bool gateIsOff,
                           V4 openTh, V4 closeTh, V4 gC) noexcept
{
    const V4 one = V4::splat(1.f);
    const V4 zero = V4::splat(0.f);
    if (gateIsOff)
    {
        open = one;
        gate = gate + gC * (one - gate);
        gate = vsel(vgt(gate, V4::splat(0.999f)), one, gate);
        return;
    }
    const Mask isOpen = vgt(open, V4::splat(0.5f));
    const V4 stay = vsel(vlt(env, closeTh), zero, one);
    const V4 fromClosed = vsel(vge(env, openTh), one, zero);
    open = vsel(isOpen, stay, fromClosed);
    gate = gate + gC * (open - gate);
    gate = vsel(vlt(gate, V4::splat(1.0e-7f)), zero, vmax(gate, zero));
}

VOC_INLINE V4 enhBoost(V4 rms, float flo) noexcept
{
    const V4 one = V4::splat(1.f);
    const float span = flo * (kEnhKnee - 1.f);
    V4 t = (rms - V4::splat(flo)) * V4::splat(1.f / span);
    t = vmin(vmax(t, V4::splat(0.f)), one);
    const V4 s = t * t * (V4::splat(3.f) - V4::splat(2.f) * t);
    const V4 b = V4::splat(kNominal) / vmax(rms, V4::splat(1.0e-12f));
    return b * s;
}

float polyblep(float t, float dt) noexcept
{
    if (dt <= 1.0e-12f)
        return 0.f;
    if (t < dt)
    {
        t /= dt;
        return t + t - t * t - 1.f;
    }
    if (t > 1.f - dt)
    {
        t = (t - 1.f) / dt;
        return t * t + t + t + 1.f;
    }
    return 0.f;
}
} // namespace

void VocoderEngine::prepare(double sampleRate, int)
{
    sr = sampleRate > 1000.0 ? sampleRate : 48000.0;
    silenceNeed = std::max(64, (int) std::floor(0.05 * sr));
    decim = 1;
    while (sr / (double) (decim + 1) >= 7500.0)
        ++decim;
    srDec = (float) (sr / (double) decim);
    const float lpFc = std::min(4200.f, 0.40f * srDec);
    const SvfC lp = svfSet(lpFc, 0.7071, sr);
    lpA1 = (float) lp.a1;
    lpA2 = (float) lp.a2;
    lpA3 = (float) lp.a3;
    lpK = (float) lp.k;
    hopN = std::max(16, (int) std::floor(0.004 * srDec));
    const double g = std::exp(-50.0 * kDb2A);
    yinGate2 = (float) (g * g);
    key = {};
    hasCachedParams = false;
    cachedSr = -1.0;
    reset();

    // Pre-build initial default filterbank so first audio block does less work
    designTargets();
    smoothCoeffs(kVocSub, true);
}

void VocoderEngine::reset()
{
    boot = true;
    coefSettling = true;
    asleep = false;
    devCleared = false;
    silenceHold = 0;
    const V4 z = V4::splat(0.f);
    const V4 one = V4::splat(1.f);
    for (int i = 0; i < kVocMaxBands; ++i)
    {
        bands[(size_t) i].initialized = false;
        bands[(size_t) i].sign = (i & 1) ? -1.0 : 1.0;
    }

    for (auto& Q : quads)
    {
        Q = Quad {};
        Q.al1 = Q.al2 = Q.al3 = Q.al4 = z;
        Q.ar1 = Q.ar2 = Q.ar3 = Q.ar4 = z;
        Q.sl1 = Q.sl2 = Q.sl3 = Q.sl4 = z;
        Q.sr1 = Q.sr2 = Q.sr3 = Q.sr4 = z;
        Q.envL = Q.envR = z;
        Q.enhL = Q.enhR = z;
        Q.gateL = Q.gateR = Q.openL = Q.openR = one;
        Q.att = Q.attTarget = one;
    }
    formantSm = 0.f;
    formantApplied = 1.0e9f;
    depthSm = depthTarget = 100.f;
    depthMode = 1;
    depthK = 1.f;
    lvl = wet = dev = duck = 1.f;
    enhMix = 0.f;
    enhPeak = 0.f;
    uvEnv = 0.f;
    activeCarrier = -1;
    activeCh = -1;
    pending = false;
    duckTarget = 1.f;
    shPos = shL = shR = 0.f;
    oscPhase = 0.f;
    ptTarget = 100.f;
    ptTargetLog = std::log(100.f);
    ptLog = ptTargetLog;
    decA1 = decA2 = decB1 = decB2 = 0.f;
    decI = hopCount = ringW = ringFill = 0;
    ring.fill(0.f);
    designHoldSamples = 0;
    designPending = false;
    prevActiveBands = -1;
    key = {};
    hasCachedParams = false;
    cachedSr = -1.0;
}

void VocoderEngine::clearSynthIC()
{
    const V4 z = V4::splat(0.f);
    for (auto& Q : quads)
    {
        Q.sl1 = Q.sl2 = Q.sl3 = Q.sl4 = z;
        Q.sr1 = Q.sr2 = Q.sr3 = Q.sr4 = z;
        Q.enhL = Q.enhR = z;
    }
}

void VocoderEngine::clearAnalysisIC(bool clearEnvelopes)
{
    const V4 z = V4::splat(0.f);
    const V4 one = V4::splat(1.f);
    for (auto& Q : quads)
    {
        Q.al1 = Q.al2 = Q.al3 = Q.al4 = z;
        Q.ar1 = Q.ar2 = Q.ar3 = Q.ar4 = z;
        if (clearEnvelopes)
        {
            Q.envL = Q.envR = z;
            Q.gateL = Q.gateR = Q.openL = Q.openR = one;
        }
    }
}

void VocoderEngine::sanitizeFilterStates()
{
    healScalar(lvl, lvlTarget);
    healScalar(wet, wetTarget);
    healScalar(dev, devTarget);
    healScalar(enhMix, enhTarget);
    healScalar(depthSm, depthTarget);
    healScalar(depthBlend, 1.f);
    healScalar(formantSm, formantTarget);
    healScalar(duck, duckTarget);
    healScalar(uvEnv, 0.f);
    if (!vocIsFinite(enhPeak))
        enhPeak = 0.f;
    if (!vocIsFinite(decA1) || !vocIsFinite(decA2) || !vocIsFinite(decB1) || !vocIsFinite(decB2))
        decA1 = decA2 = decB1 = decB2 = 0.f;
    healScalar(ptLog, ptTargetLog);
    if (!vocIsFinite(oscPhase) || oscPhase < 0.f || oscPhase >= 1.f)
        oscPhase = 0.f;

    bool ok = true;
    for (int q = 0; q < kVocQuads; ++q)
    {
        const Quad& Q = quads[(size_t) q];
        if (!Q.al1.allFinite() || !Q.al2.allFinite() || !Q.al3.allFinite() || !Q.al4.allFinite()
            || !Q.ar1.allFinite() || !Q.ar2.allFinite() || !Q.ar3.allFinite() || !Q.ar4.allFinite()
            || !Q.sl1.allFinite() || !Q.sl2.allFinite() || !Q.sl3.allFinite() || !Q.sl4.allFinite()
            || !Q.sr1.allFinite() || !Q.sr2.allFinite() || !Q.sr3.allFinite() || !Q.sr4.allFinite()
            || !Q.envL.allFinite() || !Q.envR.allFinite() || !Q.enhL.allFinite() || !Q.enhR.allFinite())
        {
            ok = false;
            break;
        }
    }
    if (!ok || !(enhPeak >= 0.f && enhPeak < 1.0e12f) || !(uvEnv >= 0.f && uvEnv <= 2.f))
    {
        clearAnalysisIC(true);
        clearSynthIC();
        enhPeak = 0.f;
        uvEnv = 0.f;
    }
}

void VocoderEngine::clearBandLane(int band) noexcept
{
    Quad& Q = quads[(size_t) (band >> 2)];
    const int lane = band & 3;
    auto setLane = [lane](V4& v, float value) {
        alignas(16) float t[4];
        v.store(t);
        t[lane] = value;
        v = V4::load(t);
    };
    setLane(Q.al1, 0.f);
    setLane(Q.al2, 0.f);
    setLane(Q.al3, 0.f);
    setLane(Q.al4, 0.f);
    setLane(Q.ar1, 0.f);
    setLane(Q.ar2, 0.f);
    setLane(Q.ar3, 0.f);
    setLane(Q.ar4, 0.f);
    setLane(Q.sl1, 0.f);
    setLane(Q.sl2, 0.f);
    setLane(Q.sl3, 0.f);
    setLane(Q.sl4, 0.f);
    setLane(Q.sr1, 0.f);
    setLane(Q.sr2, 0.f);
    setLane(Q.sr3, 0.f);
    setLane(Q.sr4, 0.f);
    setLane(Q.envL, 0.f);
    setLane(Q.envR, 0.f);
    setLane(Q.enhL, 0.f);
    setLane(Q.enhR, 0.f);
    setLane(Q.gateL, 1.f);
    setLane(Q.gateR, 1.f);
    setLane(Q.openL, 1.f);
    setLane(Q.openR, 1.f);
}

void VocoderEngine::retireInactiveBands() noexcept
{
    // Bands that have faded out (or were dropped before the 12 ms mix tail
    // reached zero) are no longer visited by analyze/synth. Their envelopes
    // and SVF integrators would otherwise stay frozen and blast on the next
    // Bands increase. A lane that is switching back on from that silence
    // starts at zero so the attack rebuilds it.
    const int prev = prevActiveBands;
    for (int i = 0; i < kVocMaxBands; ++i)
    {
        Quad& Q = quads[(size_t) (i >> 2)];
        const int lane = i & 3;
        alignas(16) float mix[4];
        Q.mix.store(mix);
        const float m = mix[lane];
        const bool active = i < nBands;
        const bool reactivated = active && prev >= 0 && i >= prev && m <= 0.001f;
        const bool faded = !active && m <= 0.001f;
        if (!reactivated && !faded)
            continue;
        if (faded && m == 0.f)
        {
            alignas(16) float env[4], ic[4], envR[4], icR[4];
            Q.envL.store(env);
            Q.envR.store(envR);
            Q.al1.store(ic);
            Q.sl1.store(icR);
            if (env[lane] == 0.f && envR[lane] == 0.f && ic[lane] == 0.f && icR[lane] == 0.f)
                continue;
        }
        clearBandLane(i);
        Q.mix.store(mix);
        mix[lane] = 0.f;
        Q.mix = V4::load(mix);
    }
    prevActiveBands = nBands;
}

void VocoderEngine::resetPitchRing()
{
    ring.fill(0.f);
    ringW = 0;
    ringFill = 0;
    hopCount = 0;
    decI = 0;
    decA1 = decA2 = decB1 = decB2 = 0.f;
}

void VocoderEngine::designTargets()
{
    double lo = std::max(20.0, (double) flo);
    double hi = std::min(sr * 0.45, (double) fhi);
    if (hi < lo * 1.25)
        hi = std::min(lo * 1.25, sr * 0.45);
    if (hi <= lo)
        hi = std::min(lo * 2.0, sr * 0.45);
    const double span = std::log(hi / lo);
    double bwOct = nBands > 1
        ? (span / kLn2) / (double) (nBands - 1) * ((double) bwPct / 100.0)
        : 1.0;
    if (bwOct < 0.02)
        bwOct = 0.02;

    auto qOf = [&](double fc, double oct) {
        double r = std::exp(oct * 0.5 * kLn2);
        double fe = fc * r;
        if (fe > sr * 0.49)
            fe = sr * 0.49;
        double oc = std::tan(kPi * fc / sr);
        double oe = std::tan(kPi * fe / sr);
        if (oc < 1.0e-12)
            oc = 1.0e-12;
        r = oe / oc;
        double qq = r / (std::abs(1.0 - r * r) * kQDen + 1.0e-12);
        return std::clamp(qq, 0.12, 64.0);
    };

    auto h4 = [&](double freq, double fc, double qq) {
        double g = std::tan(kPi * fc / sr);
        if (g < 1.0e-12)
            g = 1.0e-12;
        const double k = 1.0 / qq;
        const double w = std::tan(kPi * std::min(freq, sr * 0.499) / sr) / g;
        double den = 1.0 - w * w;
        den = den * den + (k * w) * (k * w);
        if (den < 1.0e-18)
            den = 1.0e-18;
        double num = k * k * w;
        num = num * num * (1.0 + w * w) / (2.0 * den * den);
        return num;
    };

    auto bandVr = [&](double fc, double qq) {
        const double g = std::tan(kPi * std::min(fc, sr * 0.45) / sr);
        const double analytic = g / (2.0 * qq * (1.0 + g * g));
        const double ratio = fc / sr;
        // Below 0.08*sr the closed form matches the old early-out (which used
        // 0.12*sr). High bands, and the crossfade up to 0.14*sr, integrate on
        // a grid that always lands on fc so a narrow high-Q peak cannot fall
        // between 0.14-octave steps.
        if (qq >= 1.2 && ratio <= 0.08)
            return std::max(analytic, 1.0e-10);

        int n = (int) std::ceil(72.0 * std::max(1.0, qq / 4.0));
        n = std::clamp(n, 144, 1024);
        if (n & 1)
            n = n < 1024 ? n + 1 : n - 1;
        const double fMin = std::max(fc / 32.0, 1.0);
        const double fMax = std::min(fc * 32.0, sr * 0.49);
        double numeric = analytic;
        if (fMax > fMin)
        {
            const double fcSplit = std::clamp(fc, fMin, fMax);
            const int half = std::max(1, n / 2);
            auto trap = [&](double fa, double fb, int steps) {
                if (steps < 1 || !(fb > fa))
                    return 0.0;
                const double logStep = std::log(fb / fa) / (double) steps;
                double prevF = fa;
                double prevP = h4(fa, fc, qq);
                double acc = 0.0;
                for (int i = 1; i <= steps; ++i)
                {
                    const double f = fa * std::exp(logStep * (double) i);
                    const double p = h4(f, fc, qq);
                    acc += (prevP + p) * 0.5 * (f - prevF);
                    prevF = f;
                    prevP = p;
                }
                return acc;
            };
            numeric = 2.0 * (trap(fMin, fcSplit, half) + trap(fcSplit, fMax, half)) / sr;
        }
        double t = (ratio - 0.08) / (0.14 - 0.08);
        t = std::clamp(t, 0.0, 1.0);
        t = t * t * (3.0 - 2.0 * t);
        if (qq < 1.2 || ratio >= 0.14)
            t = 1.0;
        return std::max(analytic * (1.0 - t) + numeric * t, 1.0e-10);
    };

    for (int i = 0; i < kVocMaxBands; ++i)
    {
        Band& b = bands[(size_t) i];
        b.sign = (i & 1) ? -1.0 : 1.0;
        if (i < nBands)
        {
            const double t = nBands > 1 ? (double) i / (double) (nBands - 1) : 0.0;
            double fc = std::exp(std::log(lo) + span * (nBands > 1 ? (double) i / (double) (nBands - 1) : 0.5));
            if (fc > sr * 0.45)
                fc = sr * 0.45;
            double qq = qOf(fc, bwOct);
            if (retro)
            {
                qq = std::min(64.0, qq * (1.0 + t * 1.5));
                b.retroT = std::exp(t * 6.0 * kDb2A);
            }
            else
            {
                b.retroT = 1.0;
            }
            b.logFcT = std::log(fc);
            b.qT = qq;
            // Reference the noise-gain law to 48 kHz. At 48000 this multiply
            // is exactly 1, so the calibrated trims stay bit-identical.
            const double vr = std::max(bandVr(fc, qq) * (sr / 48000.0), 1.0e-10);
            b.invSqrtVrT = 1.0 / std::sqrt(vr);
            b.fc = fc;
            if (!b.initialized)
            {
                b.logFc = b.logFcT;
                b.q = b.qT;
                b.retro = b.retroT;
                b.invSqrtVr = b.invSqrtVrT;
            }
        }
        else if (!b.initialized)
        {
            b.logFc = b.logFcT = std::log(lo);
            b.q = b.qT = 1.0;
            b.retro = b.retroT = 1.0;
            b.flat = b.flatT = 0.0;
            b.invSqrtVr = b.invSqrtVrT = 1.0;
            b.fc = lo;
        }
        // Previously active bands (i >= nBands && b.initialized) keep their previous logFcT, qT,
        // invSqrtVrT, flatT, retroT so that while their `mix` fades to 0 over 12 ms, they do not
        // chirp down to `lo`, flip polarity, or jump in gain.
    }

    constexpr int nfreq = kVocDesignFreqs;
    double fmin = lo * 1.25;
    double fmax = hi * 0.8;
    if (fmax > sr * 0.45)
        fmax = sr * 0.45;
    if (nBands < 2 || fmax <= fmin * 1.02)
    {
        for (int i = 0; i < nBands; ++i)
        {
            bands[(size_t) i].flatT = 1.0;
            if (!bands[(size_t) i].initialized)
            {
                bands[(size_t) i].flat = 1.0;
                bands[(size_t) i].initialized = true;
            }
        }
        coefSettling = true;
        return;
    }

    for (int bi = 0; bi < nBands; ++bi)
        solveG[(size_t) bi] = 1.0;

    auto resp = [&](double freq, double fc, double qq, double& re, double& im) {
        double g = std::tan(kPi * std::min(fc, sr * 0.45) / sr);
        if (g < 1.0e-12)
            g = 1.0e-12;
        const double k = 1.0 / std::max(qq, 0.12);
        const double w = std::tan(kPi * std::min(freq, sr * 0.499) / sr) / g;
        const double dr = 1.0 - w * w;
        const double di = k * w;
        const double ni = k * w;
        double den = dr * dr + di * di;
        if (den < 1.0e-18)
            den = 1.0e-18;
        const double pr = (ni * di) / den;
        const double pj = (ni * dr) / den;
        re = pr * pr - pj * pj;
        im = 2.0 * pr * pj;
    };

    const double log0 = std::log(fmin);
    const double log1 = std::log(fmax);
    for (int fi = 0; fi < nfreq; ++fi)
    {
        const double freq = std::exp(log0 + (log1 - log0) * (double) fi / (double) (nfreq - 1));
        for (int bi = 0; bi < nBands; ++bi)
        {
            const Band& b = bands[(size_t) bi];
            double re = 0, im = 0;
            resp(freq, std::exp(b.logFcT), b.qT, re, im);
            const size_t idx = (size_t) fi * kVocMaxBands + (size_t) bi;
            solveHR[idx] = re * b.sign;
            solveHI[idx] = im * b.sign;
        }
    }

    auto medianInPlace = [&]() -> double {
        std::memcpy(solveTmp.data(), solveMag.data(), sizeof(double) * nfreq);
        const size_t mid = nfreq / 2;
        std::nth_element(solveTmp.begin(), solveTmp.begin() + (ptrdiff_t) mid, solveTmp.begin() + nfreq);
        const double hiMed = solveTmp[mid];
        double loMed = solveTmp[0];
        for (size_t i = 1; i < mid; ++i)
            loMed = std::max(loMed, solveTmp[i]);
        return 0.5 * (loMed + hiMed);
    };

    for (int iter = 0; iter < 4; ++iter)
    {
        for (int fi = 0; fi < nfreq; ++fi)
        {
            double ar = 0, ai = 0;
            for (int bi = 0; bi < nBands; ++bi)
            {
                const size_t idx = (size_t) fi * kVocMaxBands + (size_t) bi;
                ar += solveHR[idx] * solveG[(size_t) bi];
                ai += solveHI[idx] * solveG[(size_t) bi];
            }
            solveSR[(size_t) fi] = ar;
            solveSI[(size_t) fi] = ai;
            solveMag[(size_t) fi] = std::sqrt(ar * ar + ai * ai) + 1.0e-12;
        }
        const double med = medianInPlace();
        for (int bi = 0; bi < nBands; ++bi)
        {
            double num = 0, den = 0;
            for (int fi = 0; fi < nfreq; ++fi)
            {
                const size_t idx = (size_t) fi * kVocMaxBands + (size_t) bi;
                const double xr = solveHR[idx];
                const double xi = solveHI[idx];
                const double m = solveMag[(size_t) fi];
                const double tr = med * solveSR[(size_t) fi] / m - solveSR[(size_t) fi] + solveG[(size_t) bi] * xr;
                const double ti = med * solveSI[(size_t) fi] / m - solveSI[(size_t) fi] + solveG[(size_t) bi] * xi;
                num += xr * tr + xi * ti;
                den += xr * xr + xi * xi;
            }
            if (den < 1.0e-10)
                den = 1.0e-10;
            double gn = num / den;
            gn = 0.55 * solveG[(size_t) bi] + 0.45 * gn;
            solveG[(size_t) bi] = std::clamp(gn, 0.2, 5.0);
        }
    }

    for (int fi = 0; fi < nfreq; ++fi)
    {
        double ar = 0, ai = 0;
        for (int bi = 0; bi < nBands; ++bi)
        {
            const size_t idx = (size_t) fi * kVocMaxBands + (size_t) bi;
            ar += solveHR[idx] * solveG[(size_t) bi];
            ai += solveHI[idx] * solveG[(size_t) bi];
        }
        solveMag[(size_t) fi] = std::sqrt(ar * ar + ai * ai) + 1.0e-12;
    }
    double med = medianInPlace();
    if (med < 1.0e-4)
        med = 1.0;
    for (int bi = 0; bi < nBands; ++bi)
    {
        bands[(size_t) bi].flatT = solveG[(size_t) bi] / med;
        if (!bands[(size_t) bi].initialized)
        {
            bands[(size_t) bi].flat = bands[(size_t) bi].flatT;
            bands[(size_t) bi].initialized = true;
        }
    }

    coefSettling = true;
}

void VocoderEngine::smoothCoeffs(int stepSamples, bool snap)
{
    const double a = snap ? 1.0 : std::min(1.0, 1.0 - std::exp(-(double) stepSamples / (0.02 * sr)));
    bool moved = snap;
    bool svfMoved = snap;

    if (snap)
    {
        formantSm = formantTarget;
    }
    else
    {
        const double dF = (double) formantTarget - (double) formantSm;
        if (std::abs(dF) <= 1.0e-4)
        {
            if (std::memcmp(&formantSm, &formantTarget, sizeof(float)) != 0)
            {
                formantSm = formantTarget;
                moved = svfMoved = true;
            }
        }
        else
        {
            const double fa = 1.0 - std::exp(-(double) stepSamples / (0.012 * sr));
            formantSm += (float) (fa * dF);
            moved = svfMoved = true;
        }
    }
    const double formRatio = std::exp((double) formantSm * kLn2 / 12.0);

    bool anyUnsettled = (std::memcmp(&formantSm, &formantTarget, sizeof(float)) != 0);
    for (int i = 0; i < kVocMaxBands; ++i)
    {
        Band& b = bands[(size_t) i];
        if (snap)
        {
            b.logFc = b.logFcT;
            b.q = b.qT;
            b.retro = b.retroT;
            b.flat = b.flatT;
            b.invSqrtVr = b.invSqrtVrT;
        }
        else
        {
            auto stepVal = [&](double& cur, double tgt, double eps, bool isSvf) {
                const double d = tgt - cur;
                if (std::abs(d) <= eps)
                {
                    if (std::memcmp(&cur, &tgt, sizeof(double)) != 0)
                    {
                        cur = tgt;
                        moved = true;
                        if (isSvf)
                            svfMoved = true;
                    }
                }
                else
                {
                    cur += a * d;
                    moved = true;
                    anyUnsettled = true;
                    if (isSvf)
                        svfMoved = true;
                }
            };
            stepVal(b.logFc, b.logFcT, 1.0e-5, true);
            stepVal(b.q, b.qT, 1.0e-5, true);
            stepVal(b.retro, b.retroT, 1.0e-5, false);
            stepVal(b.flat, b.flatT, 1.0e-5, false);
            stepVal(b.invSqrtVr, b.invSqrtVrT, 1.0e-4, false);
        }
    }

    if (svfMoved || std::abs(formantSm - formantApplied) > 1.0e-4f)
    {
        for (int i = 0; i < kVocMaxBands; ++i)
        {
            Band& b = bands[(size_t) i];
            const double fc = std::clamp(std::exp(b.logFc), 1.0, sr * 0.45);
            const double qq = std::max(b.q, 0.12);
            const SvfC an = svfSet(fc, qq, sr);
            b.a1 = (float) an.a1;
            b.a2 = (float) an.a2;
            b.a3 = (float) an.a3;
            b.k = (float) an.k;
            const double fcs = std::clamp(fc * formRatio, 1.0, sr * 0.45);
            const SvfC sy = svfSet(fcs, qq, sr);
            b.sa1 = (float) sy.a1;
            b.sa2 = (float) sy.a2;
            b.sa3 = (float) sy.a3;
            b.sk = (float) sy.k;
            b.fc = fc;
        }
        formantApplied = formantSm;
        moved = true;
    }

    if (moved)
        packQuads();

    retireInactiveBands();
    coefSettling = anyUnsettled;
}

void VocoderEngine::packQuads()
{
    for (int q = 0; q < kVocQuads; ++q)
    {
        alignas(16) float a1[4], a2[4], a3[4], k[4];
        alignas(16) float s1[4], s2[4], s3[4], sk[4];
        alignas(16) float inv[4], pol[4], flat[4], retArr[4], hi[4];
        for (int i = 0; i < 4; ++i)
        {
            const Band& b = bands[(size_t) (q * 4 + i)];
            a1[i] = b.a1; a2[i] = b.a2; a3[i] = b.a3; k[i] = b.k;
            s1[i] = b.sa1; s2[i] = b.sa2; s3[i] = b.sa3; sk[i] = b.sk;
            inv[i] = (float) b.invSqrtVr;
            pol[i] = (float) b.sign;
            flat[i] = (float) b.flat;
            retArr[i] = (float) b.retro;
            hi[i] = b.fc >= 4000.0 ? 1.f : 0.f;
        }
        Quad& Q = quads[(size_t) q];
        Q.a1 = V4::load(a1); Q.a2 = V4::load(a2); Q.a3 = V4::load(a3); Q.k = V4::load(k);
        Q.sa1 = V4::load(s1); Q.sa2 = V4::load(s2); Q.sa3 = V4::load(s3); Q.sk = V4::load(sk);
        Q.invVr = V4::load(inv);
        Q.pol = V4::load(pol);
        Q.flat = V4::load(flat);
        Q.retro = V4::load(retArr);
        Q.hi = V4::load(hi);
    }
}

void VocoderEngine::updateActiveMask()
{
    for (int q = 0; q < kVocQuads; ++q)
    {
        alignas(16) float a[4];
        for (int i = 0; i < 4; ++i)
            a[i] = (q * 4 + i) < nBands ? 1.f : 0.f;
        quads[(size_t) q].active = V4::load(a);
    }
}

void VocoderEngine::computeRunQuads()
{
    retireInactiveBands();
    nRunQuads = 0;
    for (int q = 0; q < kVocQuads; ++q)
    {
        alignas(16) float mix[4], act[4];
        quads[(size_t) q].mix.store(mix);
        quads[(size_t) q].active.store(act);
        bool run = false;
        for (int i = 0; i < 4; ++i)
            if (mix[i] > 0.001f || act[i] > 0.5f)
                run = true;
        if (run)
            nRunQuads = q + 1;
    }
    if (nRunQuads < 1)
        nRunQuads = 1;
}

void VocoderEngine::considerRoute()
{
    if (wantCarrier != activeCarrier || wantCh != activeCh)
    {
        pendingCarrier = wantCarrier;
        pendingCh = wantCh;
        pending = true;
        duckTarget = 0.f;
    }
    else
    {
        // If user switched A -> B -> A before duck landed, cancel pending transition
        pending = false;
        duckTarget = 1.f;
    }

    if (pending && duck < 0.02f)
    {
        const int prevCh = activeCh;
        const int prevCar = activeCarrier;
        activeCarrier = pendingCarrier;
        activeCh = pendingCh;
        pending = false;
        duckTarget = 1.f;
        clearSynthIC();
        // When entering L/R mode from Mono/Stereo, copy left analysis states/envelopes into right
        // instead of leaving stale envR or zeroing al1..al4 on Mono <-> Stereo switches.
        if (activeCh == 2 && prevCh != 2)
        {
            for (auto& Q : quads)
            {
                Q.ar1 = Q.al1;
                Q.ar2 = Q.al2;
                Q.ar3 = Q.al3;
                Q.ar4 = Q.al4;
                Q.envR = Q.envL;
                Q.openR = Q.openL;
                Q.gateR = Q.gateL;
            }
        }
        if (activeCarrier == 3 && prevCar != 3)
            resetPitchRing();
    }
}

void VocoderEngine::smoothMacros(int m)
{
    for (int n = 0; n < m; ++n)
    {
        duck += cDuck * (duckTarget - duck);
        lvl += cMacro * (lvlTarget - lvl);
        wet += cMacro * (wetTarget - wet);
        dev += cMacro * (devTarget - dev);
        enhMix += cMacro * (enhTarget - enhMix);
        duck = std::clamp(duck, 0.f, 1.f);
        wet = std::clamp(wet, 0.f, 1.f);
        dev = std::clamp(dev, 0.f, 1.f);
        enhMix = std::clamp(enhMix, 0.f, 1.f);
        duckArr[n] = duck;
        lvlArr[n] = lvl;
        wetArr[n] = wet;
        devArr[n] = dev;
        enhArr[n] = enhMix;
    }
}

float VocoderEngine::randUni()
{
    uint32_t x = rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    if (x == 0)
        x = 0xA3C59AC3u;
    rng = x;
    return (float) (x >> 8) * (1.f / 16777216.f);
}

float VocoderEngine::oscSample()
{
    ptLog += cGlide * (ptTargetLog - ptLog);
    float dt = vocFastExp(ptLog) * ptTranspose / (float) sr;
    dt = std::clamp(dt, 0.f, 0.45f);
    oscPhase += dt;
    if (oscPhase >= 1.f)
        oscPhase -= 1.f;

    float osc = 0.f;
    if (oscWave <= 0)
    {
        osc = (2.f * oscPhase - 1.f) - polyblep(oscPhase, dt);
        osc *= kSaw;
    }
    else
    {
        const float pw = oscWave == 1 ? 0.5f : oscWave == 2 ? 0.25f : 0.125f;
        osc = oscPhase < pw ? 1.f : -1.f;
        osc += polyblep(oscPhase, dt);
        float ph2 = oscPhase + 1.f - pw;
        if (ph2 >= 1.f)
            ph2 -= 1.f;
        osc -= polyblep(ph2, dt);
        osc -= (pw + pw - 1.f);
        osc *= oscWave == 1 ? kP50 : oscWave == 2 ? kP25 : kP125;
    }
    return osc;
}

void VocoderEngine::yinRun()
{
    int tauMax = (int) std::floor(srDec / std::max(tLow, 20.f));
    tauMax = std::clamp(tauMax, 8, 1500);
    int tauMin = (int) std::floor(srDec / std::max(tHigh, tLow + 1.f));
    tauMin = std::max(2, tauMin);
    if (tauMin >= tauMax - 2)
        tauMin = std::max(2, tauMax - 4);
    const int w = tauMax + 2;
    const int frame = w + w;
    if (ringFill < frame || frame > (int) ybuf.size() || tauMax + 2 >= (int) yinD.size())
        return;

    int start = ringW - frame;
    if (start < 0)
        start += kRing;
    for (int j = 0; j < frame; ++j)
    {
        int idx = start + j;
        if (idx >= kRing)
            idx -= kRing;
        ybuf[(size_t) j] = ring[(size_t) idx];
    }

#if defined(__APPLE__)
    float mean = 0.f;
    vDSP_meanv(ybuf.data(), 1, &mean, (vDSP_Length) frame);
    const float neg = -mean;
    vDSP_vsadd(ybuf.data(), 1, &neg, ybuf.data(), 1, (vDSP_Length) frame);
    float energy = 0.f;
    vDSP_svesq(ybuf.data(), 1, &energy, (vDSP_Length) frame);
#else
    float mean = 0.f;
    for (int j = 0; j < frame; ++j)
        mean += ybuf[(size_t) j];
    mean /= (float) frame;
    float energy = 0.f;
    for (int j = 0; j < frame; ++j)
    {
        ybuf[(size_t) j] -= mean;
        energy += ybuf[(size_t) j] * ybuf[(size_t) j];
    }
#endif
    if (energy < yinGate2 * (float) frame)
        return;

    float s1 = 0.f;
#if defined(__APPLE__)
    vDSP_svesq(ybuf.data(), 1, &s1, (vDSP_Length) w);
    float s2 = 0.f;
    vDSP_svesq(ybuf.data() + 1, 1, &s2, (vDSP_Length) w);
#else
    for (int j = 0; j < w; ++j)
        s1 += ybuf[(size_t) j] * ybuf[(size_t) j];
    float s2 = 0.f;
    for (int j = 0; j < w; ++j)
        s2 += ybuf[(size_t) (j + 1)] * ybuf[(size_t) (j + 1)];
#endif

    float acc = 0.f;
    yinD[0] = 1.f;
    int best = tauMin;
    float bestv = 1.0e9f;
    for (int tau = 1; tau <= tauMax; ++tau)
    {
        if (tau > 1)
        {
            const float leave = ybuf[(size_t) (tau - 1)];
            const float enter = ybuf[(size_t) (tau - 1 + w)];
            s2 += enter * enter - leave * leave;
            if (s2 < 0.f)
                s2 = 0.f;
        }
        float rr = 0.f;
#if defined(__APPLE__)
        vDSP_dotpr(ybuf.data(), 1, ybuf.data() + tau, 1, &rr, (vDSP_Length) w);
#else
        for (int j = 0; j < w; ++j)
            rr += ybuf[(size_t) j] * ybuf[(size_t) (j + tau)];
#endif
        float dd = s1 + s2 - 2.f * rr;
        if (dd < 0.f)
            dd = 0.f;
        acc += dd;
        const float cm = acc > 0.f ? (dd * (float) tau / acc) : 1.f;
        yinD[(size_t) tau] = cm;
        if (tau >= tauMin && cm < bestv)
        {
            bestv = cm;
            best = tau;
        }
    }

    {
        const int tau = tauMax + 1;
        const float leave = ybuf[(size_t) (tau - 1)];
        const float enter = ybuf[(size_t) (tau - 1 + w)];
        s2 += enter * enter - leave * leave;
        if (s2 < 0.f)
            s2 = 0.f;
        float rr = 0.f;
#if defined(__APPLE__)
        vDSP_dotpr(ybuf.data(), 1, ybuf.data() + tau, 1, &rr, (vDSP_Length) w);
#else
        for (int j = 0; j < w; ++j)
            rr += ybuf[(size_t) j] * ybuf[(size_t) (j + tau)];
#endif
        float dd = s1 + s2 - 2.f * rr;
        if (dd < 0.f)
            dd = 0.f;
        acc += dd;
        yinD[(size_t) tau] = acc > 0.f ? (dd * (float) tau / acc) : 1.f;
    }

    int tau = tauMin;
    bool found = false;
    while (tau < tauMax && !found)
    {
        if (yinD[(size_t) tau] < 0.15f)
            found = true;
        else
            ++tau;
    }
    if (found)
    {
        while (tau + 1 < tauMax && yinD[(size_t) (tau + 1)] < yinD[(size_t) tau])
            ++tau;
    }
    else if (bestv > 0.40f)
    {
        tau = 0;
    }
    else
    {
        tau = best;
    }

    if (tau > 1)
    {
        const float ya = yinD[(size_t) (tau - 1)];
        const float yb = yinD[(size_t) tau];
        const float yc = yinD[(size_t) (tau + 1)];
        const float denom = ya - 2.f * yb + yc;
        float tauF = std::abs(denom) > 1.0e-12f ? ((float) tau + 0.5f * (ya - yc) / denom) : (float) tau;
        if (tauF < 1.f)
            tauF = (float) tau;
        float f0 = srDec / tauF;
        f0 = std::clamp(f0, tLow, tHigh);
        ptTarget = f0;
        ptTargetLog = std::log(std::max(f0, 1.f));
    }
}

void VocoderEngine::tickPitchSample(float mono)
{
    const float v3a = mono - decA2;
    const float v1a = lpA1 * decA1 + lpA2 * v3a;
    const float v2a = decA2 + lpA2 * decA1 + lpA3 * v3a;
    decA1 = 2.f * v1a - decA1;
    decA2 = 2.f * v2a - decA2;
    const float v3b = v2a - decB2;
    const float v1b = lpA1 * decB1 + lpA2 * v3b;
    const float v2b = decB2 + lpA2 * decB1 + lpA3 * v3b;
    decB1 = 2.f * v1b - decB1;
    decB2 = 2.f * v2b - decB2;

    if (++decI >= decim)
    {
        decI = 0;
        ring[(size_t) ringW] = v2b;
        ringW = (ringW + 1) & (kRing - 1);
        if (ringFill < 1000000)
            ++ringFill;
        if (++hopCount >= hopN)
        {
            hopCount = 0;
            yinRun();
        }
    }
}

void VocoderEngine::buildIo(int base, int m, const float* inL, const float* inR,
                            const float* extL, const float* extR)
{
    for (int n = 0; n < m; ++n)
    {
        const float iL = inL != nullptr ? vocSanitize(inL[base + n]) : 0.f;
        const float iR = inR != nullptr ? vocSanitize(inR[base + n]) : iL;
        subInL[n] = iL;
        subInR[n] = iR;
        if (activeCh == 2)
        {
            subModL[n] = iL;
            subModR[n] = iR;
        }
        else
        {
            subModL[n] = subModR[n] = 0.5f * (iL + iR);
        }

        float cL = 0.f, cR = 0.f;
        if (activeCarrier == 0)
        {
            shPos += 1.f;
            if (shPos >= holdN)
            {
                shPos -= holdN;
                if (noiseCrackle >= 0.999f)
                {
                    shL = (randUni() * 2.f - 1.f) * kNoiseAmp;
                    shR = (randUni() * 2.f - 1.f) * kNoiseAmp;
                }
                else
                {
                    float p = noiseCrackle * noiseCrackle;
                    p *= p;
                    if (p < 0.0008f)
                        p = 0.0008f;
                    float amp = kWhiteRms / std::sqrt(p);
                    if (amp > 4.f)
                        amp = 4.f;
                    shL = randUni() < p ? (randUni() < 0.5f ? amp : -amp) : 0.f;
                    shR = randUni() < p ? (randUni() < 0.5f ? amp : -amp) : 0.f;
                }
            }
            cL = shL;
            cR = shR;
        }
        else if (activeCarrier == 1)
        {
            const float* el = extL;
            const float* er = extR != nullptr ? extR : el;
            cL = el != nullptr ? vocSanitize(el[base + n]) : 0.f;
            cR = er != nullptr ? vocSanitize(er[base + n]) : 0.f;
        }
        else if (activeCarrier == 2)
        {
            cL = iL;
            cR = iR;
        }
        else
        {
            cL = cR = oscSample();
            tickPitchSample(0.5f * (iL + iR));
        }

        if (activeCh == 0)
        {
            const float c = 0.5f * (cL + cR);
            cL = cR = c;
        }
        subCarL[n] = cL;
        subCarR[n] = cR;
    }
}

void VocoderEngine::injectUv(int m)
{
    if (uvOff && uvEnv <= 1.0e-6f)
    {
        uvEnv = 0.f;
        return;
    }
    for (int n = 0; n < m; ++n)
    {
        const float amp = uvEnv * uvLin * 0.18f;
        if (amp > 1.0e-8f)
        {
            subCarL[n] += (randUni() * 2.f - 1.f) * amp;
            subCarR[n] += (randUni() * 2.f - 1.f) * amp;
        }
        float target = 0.f;
        if (!uvOff)
        {
            const float total = eAll[n];
            const float hi = eHi[n];
            const float rms = total > 1.0e-16f ? std::sqrt(total) : 0.f;
            const float hf = total > 1.0e-16f ? hi / total : 0.f;
            if (rms > uvEnergyLin)
            {
                if (sens >= 99.9f)
                    target = 1.f;
                else
                    target = std::clamp((hf - uvThr) * 6.6666667f, 0.f, 1.f);
            }
        }
        const float c = target > uvEnv ? cUvAtt : cUvRel;
        uvEnv += c * (target - uvEnv);
        if (uvEnv < 1.0e-6f)
            uvEnv = 0.f;
    }
}

template <bool LR, bool UV>
void VocoderEngine::analyzeSub(int m)
{
    const V4 attC = V4::splat(cAtt);
    const V4 relC = V4::splat(cRel);
    const V4 bmixC = V4::splat(cBmix);
    const V4 attSmC = V4::splat(cAttSm);
    const V4 ntr = V4::splat(nTrim);
    const V4 gC = V4::splat(cGate);
    const V4 openTh = V4::splat(gateOpenTh);
    const V4 closeTh = V4::splat(gateCloseTh);
    const V4 zero = V4::splat(0.f);
    const bool gateIsOff = gateOff;

    if constexpr (UV)
    {
        for (int n = 0; n < m; ++n)
            eAll[n] = eHi[n] = 0.f;
    }

    int q = 0;
    if constexpr (!LR)
    {
        // Process two quads (8 bands) in lockstep to double ILP across the 10-cycle SVF recurrence
        for (; q + 1 < nRunQuads; q += 2)
        {
            Quad& Q0 = quads[(size_t) q];
            Quad& Q1 = quads[(size_t) (q + 1)];
            const V4 a10 = Q0.a1, a20 = Q0.a2, a30 = Q0.a3, k0 = Q0.k;
            const V4 a11 = Q1.a1, a21 = Q1.a2, a31 = Q1.a3, k1 = Q1.k;
            V4 ic10 = Q0.al1, ic20 = Q0.al2, jc10 = Q0.al3, jc20 = Q0.al4;
            V4 ic11 = Q1.al1, ic21 = Q1.al2, jc11 = Q1.al3, jc21 = Q1.al4;
            V4 env0 = Q0.envL, att0 = Q0.att, mix0 = Q0.mix, open0 = Q0.openL, gate0 = Q0.gateL;
            V4 env1 = Q1.envL, att1 = Q1.att, mix1 = Q1.mix, open1 = Q1.openL, gate1 = Q1.gateL;
            const V4 baseW0 = Q0.pol * Q0.flat * Q0.retro * ntr;
            const V4 baseW1 = Q1.pol * Q1.flat * Q1.retro * ntr;
            const V4 act0 = Q0.active, hi0 = Q0.hi, attT0 = Q0.attTarget;
            const V4 act1 = Q1.active, hi1 = Q1.hi, attT1 = Q1.attTarget;

            for (int n = 0; n < m; ++n)
            {
                const V4 x = V4::splat(subModL[n]);
                mix0 = mix0 + bmixC * (act0 - mix0);
                mix1 = mix1 + bmixC * (act1 - mix1);
                att0 = vmax(att0 + attSmC * (attT0 - att0), zero);
                att1 = vmax(att1 + attSmC * (attT1 - att1), zero);

                V4 pwr0, pwr1;
                svf2(x, a10, a20, a30, k0, ic10, ic20, jc10, jc20, pwr0);
                svf2(x, a11, a21, a31, k1, ic11, ic21, jc11, jc21, pwr1);

                env0 = followEnv(env0, pwr0, attC, relC);
                env1 = followEnv(env1, pwr1, attC, relC);
                followGate(open0, gate0, env0, gateIsOff, openTh, closeTh, gC);
                followGate(open1, gate1, env1, gateIsOff, openTh, closeTh, gC);

                if constexpr (UV)
                {
                    eAll[n] += (pwr0 + pwr1).hsum();
                    eHi[n] += (pwr0 * hi0 + pwr1 * hi1).hsum();
                }

                envHistL[q][n] = env0;
                envHistL[q + 1][n] = env1;
                wHistL[q][n] = baseW0 * att0 * mix0 * gate0;
                wHistL[q + 1][n] = baseW1 * att1 * mix1 * gate1;
            }

            Q0.al1 = ic10; Q0.al2 = ic20; Q0.al3 = jc10; Q0.al4 = jc20;
            Q0.envL = env0; Q0.att = att0; Q0.mix = mix0; Q0.gateL = gate0; Q0.openL = open0;
            Q1.al1 = ic11; Q1.al2 = ic21; Q1.al3 = jc11; Q1.al4 = jc21;
            Q1.envL = env1; Q1.att = att1; Q1.mix = mix1; Q1.gateL = gate1; Q1.openL = open1;
        }
    }

    for (; q < nRunQuads; ++q)
    {
        Quad& Q = quads[(size_t) q];
        const V4 a1 = Q.a1, a2 = Q.a2, a3 = Q.a3, k = Q.k;
        V4 ic1 = Q.al1, ic2 = Q.al2, jc1 = Q.al3, jc2 = Q.al4;
        V4 env = Q.envL, att = Q.att, mix = Q.mix, open = Q.openL, gate = Q.gateL;
        V4 ic1r = Q.ar1, ic2r = Q.ar2, jc1r = Q.ar3, jc2r = Q.ar4;
        V4 envr = Q.envR, openr = Q.openR, gater = Q.gateR;
        const V4 baseW = Q.pol * Q.flat * Q.retro * ntr;
        const V4 active = Q.active, hi = Q.hi, attT = Q.attTarget;

        for (int n = 0; n < m; ++n)
        {
            mix = mix + bmixC * (active - mix);
            att = vmax(att + attSmC * (attT - att), zero);
            const V4 wCommon = baseW * att * mix;

            V4 power;
            svf2(V4::splat(subModL[n]), a1, a2, a3, k, ic1, ic2, jc1, jc2, power);
            env = followEnv(env, power, attC, relC);
            followGate(open, gate, env, gateIsOff, openTh, closeTh, gC);
            if constexpr (UV)
            {
                const float eScale = LR ? 0.5f : 1.f;
                eAll[n] += eScale * power.hsum();
                eHi[n] += eScale * (power * hi).hsum();
            }
            envHistL[q][n] = env;
            wHistL[q][n] = wCommon * gate;

            if constexpr (LR)
            {
                V4 powerR;
                svf2(V4::splat(subModR[n]), a1, a2, a3, k, ic1r, ic2r, jc1r, jc2r, powerR);
                envr = followEnv(envr, powerR, attC, relC);
                followGate(openr, gater, envr, gateIsOff, openTh, closeTh, gC);
                if constexpr (UV)
                {
                    eAll[n] += 0.5f * powerR.hsum();
                    eHi[n] += 0.5f * (powerR * hi).hsum();
                }
                envHistR[q][n] = envr;
                wHistR[q][n] = wCommon * gater;
            }
        }

        Q.al1 = ic1;
        Q.al2 = ic2;
        Q.al3 = jc1;
        Q.al4 = jc2;
        Q.envL = env;
        Q.att = att;
        Q.mix = mix;
        Q.gateL = gate;
        Q.openL = open;
        if constexpr (LR)
        {
            Q.ar1 = ic1r;
            Q.ar2 = ic2r;
            Q.ar3 = jc1r;
            Q.ar4 = jc2r;
            Q.envR = envr;
            Q.gateR = gater;
            Q.openR = openr;
        }
    }
}

template <bool Share, bool Enhance>
void VocoderEngine::synthSub(int m)
{
    const V4 zero = V4::splat(0.f);
    const V4 enhA = V4::splat(cEnhAtt);
    const V4 enhR = V4::splat(cEnhRel);
    const bool mono = activeCh == 0;

    for (int q = 0; q < nRunQuads; ++q)
    {
        Quad& Q = quads[(size_t) q];
        const V4 a1 = Q.sa1, a2 = Q.sa2, a3 = Q.sa3, k = Q.sk;
        V4 ic1 = Q.sl1, ic2 = Q.sl2, jc1 = Q.sl3, jc2 = Q.sl4, pwr = Q.enhL;
        V4 ic1r = Q.sr1, ic2r = Q.sr2, jc1r = Q.sr3, jc2r = Q.sr4, pwrr = Q.enhR;

        for (int n = 0; n < m; ++n)
        {
            V4 ignored;
            const V4 bp = svf2(V4::splat(subCarL[n]), a1, a2, a3, k, ic1, ic2, jc1, jc2, ignored);
            if constexpr (Enhance)
            {
                const V4 bp2 = bp * bp;
                const V4 c = vsel(vgt(bp2, pwr), enhA, enhR);
                pwr = vmax(pwr + c * (bp2 - pwr), zero);
                bpHistL[q][n] = bp;
                rmsHistL[q][n] = vsqrt(pwr);
            }
            else
            {
                const float sL = (bp * gHistL[q][n]).hsum();
                subSumL[n] += sL;
                if constexpr (Share)
                {
                    if (!mono)
                        subSumR[n] += lrMode ? (bp * gHistR[q][n]).hsum() : sL;
                }
            }

            if constexpr (!Share)
            {
                const V4 bpR = svf2(V4::splat(subCarR[n]), a1, a2, a3, k, ic1r, ic2r, jc1r, jc2r, ignored);
                if constexpr (Enhance)
                {
                    const V4 bp2 = bpR * bpR;
                    const V4 c = vsel(vgt(bp2, pwrr), enhA, enhR);
                    pwrr = vmax(pwrr + c * (bp2 - pwrr), zero);
                    bpHistR[q][n] = bpR;
                    rmsHistR[q][n] = vsqrt(pwrr);
                }
                else
                {
                    subSumR[n] += (bpR * gHistR[q][n]).hsum();
                }
            }
        }

        Q.sl1 = ic1;
        Q.sl2 = ic2;
        Q.sl3 = jc1;
        Q.sl4 = jc2;
        Q.enhL = pwr;
        if constexpr (Share)
        {
            Q.sr1 = ic1;
            Q.sr2 = ic2;
            Q.sr3 = jc1;
            Q.sr4 = jc2;
            Q.enhR = pwr;
        }
        else
        {
            Q.sr1 = ic1r;
            Q.sr2 = ic2r;
            Q.sr3 = jc1r;
            Q.sr4 = jc2r;
            Q.enhR = pwrr;
        }
    }
}

void VocoderEngine::precomputeGains(int m)
{
    const V4 cap = V4::splat(kGainCap);
    const V4 zero = V4::splat(0.f);
    const V4 one = V4::splat(1.f);
    const V4 invEnh = V4::splat(kInvRefEnh);
    const V4 tiny = V4::splat(1.0e-12f);
    const float wDepth = std::clamp(depthBlend, 0.f, 1.f);
    const V4 wVec = V4::splat(wDepth);
    const V4 oneMinusW = V4::splat(1.f - wDepth);

    for (int q = 0; q < nRunQuads; ++q)
    {
        const V4 scale = quads[(size_t) q].invVr * V4::splat(kInvRefDens);
        for (int n = 0; n < m; ++n)
        {
            const float em = enhArr[n];
            auto gain = [&](V4 env) -> V4
            {
                if (gainLaw == 0)
                    return one;
                if (gainLaw == 1)
                    return vmin(env * scale, cap);
                if (gainLaw == 2)
                    return vmin(env * invEnh, cap);

                const Mask alive = vgt(env, tiny);
                const V4 bOff = vmax(env * scale, tiny);
                const V4 bOn = vmax(env * invEnh, tiny);
                V4 go, gn;
                if (depthMode == 1)
                {
                    go = vmin(bOff, cap);
                    gn = vmin(bOn, cap);
                }
                else
                {
                    const V4 kk = V4::splat(depthK);
                    go = vmin(vmax(fastExp2(kk * fastLog2(bOff)), zero), cap);
                    gn = vmin(vmax(fastExp2(kk * fastLog2(bOn)), zero), cap);
                }
                V4 g = vsel(alive, go * V4::splat(1.f - em) + gn * V4::splat(em), zero);
                if (wDepth < 1.f)
                    g = oneMinusW + wVec * g;
                return g;
            };

            gHistL[q][n] = gain(envHistL[q][n]) * wHistL[q][n];
            if (lrMode)
                gHistR[q][n] = gain(envHistR[q][n]) * wHistR[q][n];
            else
                gHistR[q][n] = gHistL[q][n];
        }
    }
}

void VocoderEngine::applyEnhance(int m, bool share)
{
    const bool mono = activeCh == 0;
    for (int n = 0; n < m; ++n)
    {
        float seen = 0.f;
        const float enhFloor = std::max(enhPeak * kEnhRel, kEnhAbs);
        const float em = enhArr[n];
        const V4 emS = V4::splat(em);
        const V4 dry = V4::splat(1.f - em);
        for (int q = 0; q < nRunQuads; ++q)
        {
            seen = std::max(seen, rmsHistL[q][n].hmax());
            const V4 bL = enhBoost(rmsHistL[q][n], enhFloor);
            const V4 mulL = em > 0.999f ? bL : dry + bL * emS;
            const V4 outVecL = bpHistL[q][n] * mulL;
            const float sL = (outVecL * gHistL[q][n]).hsum();
            subSumL[n] += sL;
            if (share)
            {
                if (!mono)
                    subSumR[n] += lrMode ? (outVecL * gHistR[q][n]).hsum() : sL;
            }
            else
            {
                seen = std::max(seen, rmsHistR[q][n].hmax());
                const V4 bR = enhBoost(rmsHistR[q][n], enhFloor);
                const V4 mulR = em > 0.999f ? bR : dry + bR * emS;
                subSumR[n] += (bpHistR[q][n] * mulR * gHistR[q][n]).hsum();
            }
        }
        const float c = seen > enhPeak ? cEnhPkAtt : cEnhPkRel;
        enhPeak += c * (seen - enhPeak);
        if (enhPeak < 0.f)
            enhPeak = 0.f;
    }
}

void VocoderEngine::mixSub(float* outL, float* outR, int base, int m)
{
    const float trim = kTrim[std::clamp(activeCarrier, 0, 3)];
    const bool mono = activeCh == 0;
    const float wDepth = depthMode == 0 ? 0.f : std::clamp(depthBlend, 0.f, 1.f);
    for (int n = 0; n < m; ++n)
    {
        float pb = 1.f;
        if (depthMode != 0)
        {
            const float pbFull = enhArr[n] + (1.f - enhArr[n]) * trim;
            pb = wDepth < 1.f ? (1.f + wDepth * (pbFull - 1.f)) : pbFull;
        }
        const float sL = subSumL[n] * pb;
        const float sR = mono ? sL : subSumR[n] * pb;
        if (!doEnhance)
            enhPeak += cEnhPkRel * (0.f - enhPeak);

        const float dL = mono ? subModL[n] : subInL[n];
        const float dR = mono ? subModL[n] : subInR[n];
        const float wetAmt = wetArr[n] * duckArr[n];
        const float dryAmt = 1.f - wetArr[n];
        const float pL = (dL * dryAmt + sL * wetAmt) * lvlArr[n];
        const float pR = (dR * dryAmt + sR * wetAmt) * lvlArr[n];
        const float devS = devArr[n];
        // Device On/Off crossfades with the true stereo input (subInL/R), so bypassing in Mono mode
        // preserves the stereo image instead of collapsing the dry path to mono.
        outL[base + n] = vocSanitize(pL * devS + subInL[n] * (1.f - devS));
        outR[base + n] = vocSanitize(pR * devS + subInR[n] * (1.f - devS));
    }
}

void VocoderEngine::publishDisp(float* disp) const
{
    if (disp == nullptr)
        return;
    for (int i = 0; i < kVocMaxBands; ++i)
        disp[i] = 0.f;
    if (asleep)
        return;
    for (int q = 0; q < nRunQuads; ++q)
    {
        alignas(16) float eL[4], eR[4], att[4];
        quads[(size_t) q].envL.store(eL);
        quads[(size_t) q].envR.store(eR);
        quads[(size_t) q].att.store(att);
        for (int i = 0; i < 4; ++i)
        {
            const int b = q * 4 + i;
            if (b < nBands)
            {
                float e = eL[i];
                if (activeCh == 2)
                    e = std::max(e, eR[i]);
                disp[b] = e * att[i];
            }
        }
    }
}

void VocoderEngine::updateDerivedParams(const VocoderParams& p, const float attDb[kVocMaxBands])
{
    const bool paramsChanged = !hasCachedParams
        || std::memcmp(&cachedSr, &sr, sizeof(double)) != 0
        || !(p == cachedParams);
    if (paramsChanged)
    {
        cachedParams = p;
        cachedSr = sr;
        hasCachedParams = true;

        wantCarrier = std::clamp(p.carrier, 0, 3);
        wantCh = std::clamp(p.channels, 0, 2);
        nBands = std::clamp(p.bands, 4, 40);
        nBands -= nBands % 4;
        if (nBands < 4)
            nBands = 4;
        flo = safeClamp(p.rangeLow, 20.f, 2000.f, flo);
        fhi = safeClamp(p.rangeHigh, 1000.f, 20000.f, fhi);
        bwPct = safeClamp(p.bw, 10.f, 200.f, bwPct);
        retro = p.filterMode != 0;
        formantTarget = safeClamp(p.formantSt, -12.f, 12.f, formantSm);
        if (std::abs(formantTarget - formantSm) > 1.0e-5f)
            coefSettling = true;

        depthTarget = safeClamp(p.depth, 0.f, 200.f, depthSm);
        if (depthTarget < 20.f)
        {
            asleep = false;
            silenceHold = 0;
        }

        const float gateDb = safeClamp(p.gateDb, -96.f, 0.f, -96.f);
        gateOff = gateDb <= -95.5f;
        const double gdb = std::max((double) gateDb, -96.0);
        gateOpenTh = (float) std::exp(gdb * kDb2A);
        gateCloseTh = (float) std::exp((gdb - 3.0) * kDb2A);

        sens = safeClamp(p.sens, 0.f, 100.f, sens);
        const float unvoicedDb = safeClamp(p.unvoicedDb, -96.f, 0.f, -96.f);
        uvOff = unvoicedDb <= -95.5f || sens <= 0.f;
        uvLin = unvoicedDb <= -95.5f ? 0.f : (float) std::exp((double) unvoicedDb * kDb2A);
        uvThr = (100.f - sens) * 0.0084f;
        uvEnergyLin = (float) std::exp(-46.0 * kDb2A);

        const float attackMs = safeClamp(p.attackMs, 0.05f, 100.f, 5.f);
        const float releaseMs = safeClamp(p.releaseMs, 0.2f, 1000.f, 80.f);
        const double atk = std::max(0.05, (double) attackMs) * 0.001;
        const double rel = std::max(0.2, (double) releaseMs) * 0.001;
        cAtt = onePole(atk, sr);
        cRel = onePole(rel, sr);
        cEnhAtt = onePole(0.004, sr);
        cEnhRel = onePole(0.015, sr);
        cEnhPkAtt = onePole(0.001, sr);
        cEnhPkRel = onePole(0.200, sr);
        cGate = onePole(0.004, sr);
        cDuck = onePole(0.0035, sr);
        cMacro = onePole(0.008, sr);
        cBmix = onePole(0.012, sr);
        cAttSm = onePole(0.008, sr);
        cGlide = onePole(0.0025, sr);
        cUvAtt = onePole(p.uvSpeed > 0 ? 0.004 : 0.012, sr);
        cUvRel = onePole(p.uvSpeed > 0 ? 0.0015 : 0.0025, sr);

        const float levelDb = safeClamp(p.levelDb, -24.f, 24.f, 0.f);
        lvlTarget = (float) std::exp((double) levelDb * kDb2A);
        const float dryWet = safeClamp(p.dryWet, 0.f, 100.f, 100.f);
        wetTarget = std::clamp(dryWet * 0.01f, 0.f, 1.f);
        devTarget = p.deviceOn ? 1.f : 0.f;
        if (!vocIsFinite(dev) || std::abs(devTarget - dev) > 1.0e-5f)
        {
            asleep = false;
            silenceHold = 0;
        }
        enhTarget = p.enhance ? 1.f : 0.f;

        noiseRate = safeClamp(p.noiseRate, 0.f, 1.f, noiseRate);
        noiseCrackle = safeClamp(p.noiseCrackle, 0.f, 1.f, noiseCrackle);
        const float rateHz = 300.f * std::exp(noiseRate * std::log((float) sr / 300.f));
        holdN = std::max(1.f, (float) sr / rateHz);

        oscWave = std::clamp(p.oscWave, 0, 3);
        ptTranspose = (float) std::exp(std::clamp(p.oscPitch, -24, 24) * kLn2 / 12.0);
        tLow = safeClamp(p.trackLow, 20.f, 1000.f, tLow);
        tHigh = std::max(tLow + 10.f, safeClamp(p.trackHigh, 100.f, 2000.f, tHigh));
        nTrim = (float) std::exp(-0.40 * std::log((double) nBands / 20.0) / kLn2 * kDb2A);
        updateActiveMask();
    }

    // Bands / Retro / sample-rate rebuild the bank immediately. Range and BW
    // are continuous, so while they are automated the solver waits until
    // ~128 samples have passed since the last design. The key stays stale
    // until then, which keeps designPending alive across blocks.
    {
        DesignKey next;
        next.bands = nBands;
        next.low = (int) std::lround((double) flo * 100.0);
        next.high = (int) std::lround((double) fhi * 100.0);
        next.bw = (int) std::lround((double) bwPct * 100.0);
        next.retro = retro ? 1 : 0;
        next.sampleRate = (int) std::lround(sr);
        if (!(next == key) || designPending)
        {
            const bool modeChanged = key.bands < 0
                || next.bands != key.bands
                || next.retro != key.retro
                || next.sampleRate != key.sampleRate;
            if (modeChanged || designHoldSamples <= 0)
            {
                designTargets();
                key = next;
                designPending = false;
                designHoldSamples = 128;
            }
            else
            {
                designPending = true;
            }
        }
    }

    if (attDb != nullptr)
    {
        bool attChanged = paramsChanged;
        if (!attChanged)
        {
            for (int i = 0; i < kVocMaxBands; ++i)
            {
                if (std::memcmp(&attDb[i], &cachedAttDb[(size_t) i], sizeof(float)) != 0)
                {
                    attChanged = true;
                    break;
                }
            }
        }
        if (attChanged)
        {
            for (int q = 0; q < kVocQuads; ++q)
            {
                alignas(16) float t[4];
                for (int i = 0; i < 4; ++i)
                {
                    const int idx = q * 4 + i;
                    cachedAttDb[(size_t) idx] = attDb[idx];
                    const float db = safeClamp(attDb[idx], -60.f, 0.f, 0.f);
                    t[i] = (float) std::exp((double) db * kDb2A);
                }
                quads[(size_t) q].attTarget = V4::load(t);
            }
        }
    }
}

void VocoderEngine::process(const float* modL, const float* modR,
                            const float* extL, const float* extR,
                            float* outL, float* outR, int numSamples,
                            const VocoderParams& p,
                            const float attDb[kVocMaxBands],
                            float dispEnv[kVocMaxBands])
{
    if (outL == nullptr || outR == nullptr || numSamples <= 0 || sr <= 0.0)
        return;

    const NoDenorms flushDenorms;

    if (designHoldSamples > 0)
        designHoldSamples = std::max(0, designHoldSamples - numSamples);

    updateDerivedParams(p, attDb);
    sanitizeFilterStates();

    if (boot)
    {
        activeCarrier = wantCarrier;
        activeCh = wantCh;
        pending = false;
        duck = duckTarget = 1.f;
        lvl = lvlTarget;
        wet = wetTarget;
        dev = devTarget;
        enhMix = enhTarget;
        depthSm = depthTarget;
        if (std::abs(depthSm - 100.f) < 0.01f)
            depthMode = 1;
        else if (depthSm < 0.01f)
            depthMode = 0;
        else
        {
            depthMode = 2;
            depthK = depthSm * 0.01f;
        }
        depthBlend = depthSm < 20.f ? std::clamp(depthSm * 0.05f, 0.f, 1.f) : 1.f;
        formantSm = formantTarget;
        const V4 one = V4::splat(1.f);
        for (auto& Q : quads)
        {
            Q.mix = Q.active;
            Q.att = Q.attTarget;
            Q.gateL = Q.gateR = Q.openL = Q.openR = one;
        }
        smoothCoeffs(kVocSub, true);
        boot = false;
    }

    // Fast path 1: Device Off settled -> true stereo bypass
    if (devTarget == 0.f && dev < 1.0e-5f)
    {
        dev = 0.f;
        if (!devCleared)
        {
            clearAnalysisIC(true);
            clearSynthIC();
            enhPeak = 0.f;
            uvEnv = 0.f;
            devCleared = true;
        }
        for (int i = 0; i < numSamples; ++i)
        {
            const float iL = modL != nullptr ? vocSanitize(modL[i]) : 0.f;
            const float iR = modR != nullptr ? vocSanitize(modR[i]) : iL;
            outL[i] = iL;
            outR[i] = iR;
        }
        if (dispEnv != nullptr)
            std::memset(dispEnv, 0, sizeof(float) * kVocMaxBands);
        return;
    }
    devCleared = false;

    // Check if modulator input is silent across the whole block
    float inPeak = 0.f;
    for (int i = 0; i < numSamples; ++i)
    {
        const float iL = modL != nullptr ? vocSanitize(modL[i]) : 0.f;
        const float iR = modR != nullptr ? vocSanitize(modR[i]) : iL;
        inPeak = std::max(inPeak, std::max(std::abs(iL), std::abs(iR)));
    }
    const bool quietIn = inPeak < 1.0e-7f;

    // Fast path 2: Idle sleep gate when modulator is silent and envelopes have decayed
    if (asleep && quietIn && depthTarget >= 20.f && depthBlend >= 1.f && depthMode != 0
        && !coefSettling && !pending && wantCarrier == activeCarrier && wantCh == activeCh
        && vocIsFinite(dev) && std::abs(devTarget - dev) < 1.0e-5f)
    {
        std::memset(outL, 0, sizeof(float) * (size_t) numSamples);
        std::memset(outR, 0, sizeof(float) * (size_t) numSamples);
        if (dispEnv != nullptr)
            std::memset(dispEnv, 0, sizeof(float) * kVocMaxBands);
        return;
    }
    if (!quietIn || depthTarget < 20.f || depthBlend < 1.f || depthMode == 0)
    {
        asleep = false;
        silenceHold = 0;
    }

    computeRunQuads();

    float outPeak = 0.f;
    int offset = 0;
    while (offset < numSamples)
    {
        const int m = std::min(kVocSub, numSamples - offset);
        if (coefSettling)
            smoothCoeffs(m, false);

        if (std::memcmp(&depthSm, &depthTarget, sizeof(float)) != 0)
        {
            const float da = 1.f - std::exp(-(float) m / (0.012f * (float) sr));
            depthSm += da * (depthTarget - depthSm);
            if (std::abs(depthTarget - depthSm) < 0.005f)
                depthSm = depthTarget;
        }
        if (std::abs(depthSm - 100.f) < 0.01f)
            depthMode = 1;
        else if (depthSm < 0.01f)
            depthMode = 0;
        else
        {
            depthMode = 2;
            depthK = depthSm * 0.01f;
        }

        const float blendTarget = depthSm < 20.f ? std::clamp(depthSm * 0.05f, 0.f, 1.f) : 1.f;
        if (std::memcmp(&depthBlend, &blendTarget, sizeof(float)) != 0)
        {
            const float ba = 1.f - std::exp(-(float) m / (0.018f * (float) sr));
            depthBlend += ba * (blendTarget - depthBlend);
            if (std::abs(blendTarget - depthBlend) < 0.0005f)
                depthBlend = blendTarget;
        }
        if (depthBlend > 0.f && depthMode == 0)
        {
            depthMode = 2;
            depthK = std::max(depthSm * 0.01f, 0.0001f);
        }

        considerRoute();
        lrMode = activeCh == 2;
        shareSynth = activeCh == 0 || (activeCarrier == 3 && uvOff && uvEnv <= 1.0e-6f);

        const bool enhOff = enhMix < 0.001f && enhTarget < 0.001f;
        const bool enhOn = enhMix > 0.999f && enhTarget > 0.999f;
        doEnhance = enhMix > 0.0005f || enhTarget > 0.0005f;
        if (depthMode == 0 && depthBlend <= 0.f)
            gainLaw = 0;
        else if (depthMode == 1 && depthBlend >= 1.f && enhOff)
            gainLaw = 1;
        else if (depthMode == 1 && depthBlend >= 1.f && enhOn)
            gainLaw = 2;
        else
            gainLaw = 3;

        std::memset(subSumL, 0, sizeof(float) * (size_t) m);
        std::memset(subSumR, 0, sizeof(float) * (size_t) m);
        smoothMacros(m);
        buildIo(offset, m, modL, modR, extL, extR);

        if (lrMode && !uvOff)
            analyzeSub<true, true>(m);
        else if (lrMode)
            analyzeSub<true, false>(m);
        else if (!uvOff)
            analyzeSub<false, true>(m);
        else
            analyzeSub<false, false>(m);

        injectUv(m);
        precomputeGains(m);

        if (doEnhance)
        {
            if (shareSynth)
                synthSub<true, true>(m);
            else
                synthSub<false, true>(m);
            applyEnhance(m, shareSynth);
        }
        else if (shareSynth)
        {
            synthSub<true, false>(m);
        }
        else
        {
            synthSub<false, false>(m);
        }

        mixSub(outL, outR, offset, m);
        for (int n = 0; n < m; ++n)
            outPeak = std::max(outPeak, std::max(std::abs(outL[offset + n]), std::abs(outR[offset + n])));
        sanitizeFilterStates();
        computeRunQuads();
        offset += m;
    }

    if (quietIn && outPeak < 1.0e-7f && depthMode != 0 && depthTarget >= 20.f && depthBlend >= 1.f)
    {
        float maxEnv = 0.f;
        for (int q = 0; q < nRunQuads; ++q)
            maxEnv = std::max(maxEnv, std::max(quads[(size_t) q].envL.hmax(), quads[(size_t) q].envR.hmax()));
        if (maxEnv < 1.0e-8f)
        {
            silenceHold += numSamples;
            if (silenceHold >= silenceNeed)
            {
                asleep = true;
                resetPitchRing();
            }
        }
        else
        {
            silenceHold = 0;
        }
    }
    else
    {
        silenceHold = 0;
    }

    publishDisp(dispEnv);
}
