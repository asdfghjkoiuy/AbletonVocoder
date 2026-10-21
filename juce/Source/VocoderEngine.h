#pragma once

#include "Simd.h"

#include <array>

// Channel vocoder. 40 bands are independent, so they are packed as 10
// struct-of-arrays quads and filtered four-at-a-time. Each quad keeps its
// SVF integrators in locals for a 32-sample sub-block (register-resident
// on ARM64 NEON). Coefficients are rebuilt only when the smoothed design
// or Formant actually moves.

static constexpr int kVocMaxBands = 40;
static constexpr int kVocQuads = 10;
static constexpr int kVocSub = 32;

struct VocoderParams
{
    int carrier = 0;     // 0 noise, 1 external, 2 modulator, 3 pitch
    int bands = 20;      // 4..40 step 4
    float rangeLow = 80.f;
    float rangeHigh = 18000.f;
    float bw = 100.f;
    int filterMode = 0;  // 0 precise, 1 retro
    float gateDb = -96.f;
    float levelDb = 0.f;
    float depth = 100.f;
    float attackMs = 5.f;
    float releaseMs = 80.f;
    float formantSt = 0.f;
    float dryWet = 100.f;
    int channels = 1;    // 0 mono, 1 stereo, 2 L/R
    bool enhance = false;
    float unvoicedDb = -96.f;
    float sens = 50.f;
    int uvSpeed = 0;     // 0 slow, 1 fast
    float noiseRate = 1.f;
    float noiseCrackle = 1.f;
    int oscWave = 0;
    int oscPitch = 0;
    float trackLow = 60.f;
    float trackHigh = 1000.f;
    bool deviceOn = true;
};

class VocoderEngine
{
public:
    void prepare(double sampleRate, int maxBlock);
    void reset();
    void process(const float* modL, const float* modR,
                 const float* extL, const float* extR,
                 float* outL, float* outR, int numSamples,
                 const VocoderParams& params,
                 const float attDb[kVocMaxBands],
                 float dispEnv[kVocMaxBands]);

private:
    struct DesignKey
    {
        int bands = -1;
        int low = 0;
        int high = 0;
        int bw = 0;
        int retro = 0;
        int sampleRate = 0;
        bool operator==(const DesignKey& o) const noexcept
        {
            return bands == o.bands && low == o.low && high == o.high
                && bw == o.bw && retro == o.retro && sampleRate == o.sampleRate;
        }
    };

    struct Band
    {
        double logFc = 0, logFcT = 0;
        double q = 1, qT = 1;
        double retro = 1, retroT = 1;
        double flat = 1, flatT = 1;
        double invSqrtVr = 1;
        double sign = 1;
        double fc = 1000;
        float a1 = 0, a2 = 0, a3 = 0, k = 1;
        float sa1 = 0, sa2 = 0, sa3 = 0, sk = 1;
    };

    struct Quad
    {
        V4 a1 {}, a2 {}, a3 {}, k {};
        V4 sa1 {}, sa2 {}, sa3 {}, sk {};
        V4 al1 {}, al2 {}, al3 {}, al4 {};
        V4 ar1 {}, ar2 {}, ar3 {}, ar4 {};
        V4 sl1 {}, sl2 {}, sl3 {}, sl4 {};
        V4 sr1 {}, sr2 {}, sr3 {}, sr4 {};
        V4 envL {}, envR {};
        V4 gateL {}, gateR {};
        V4 openL {}, openR {};
        V4 enhL {}, enhR {};
        V4 mix {}, att {}, attTarget {}, active {};
        V4 invVr {}, pol {}, flat {}, retro {}, hi {};
    };

    void designTargets();
    void smoothCoeffs(int hostBlock, bool snap);
    void packQuads();
    void updateActiveMask();
    void computeRunQuads();
    void clearSynthIC();
    void clearAnalysisIC();
    void resetPitchRing();

    template <bool LR, bool UV>
    void analyzeSub(int m);

    template <bool Share, bool Enhance>
    void synthSub(int m);

    void precomputeGains(int m);
    void applyEnhance(int m, bool share);
    void buildIo(int base, int m, const float* inL, const float* inR, const float* extL, const float* extR);
    void injectUv(int m);
    void mixSub(float* outL, float* outR, int base, int m);
    void smoothMacros(int m);
    void considerRoute();
    void tickPitchSample(float mono);
    float oscSample();
    float randUni();
    void yinRun();
    void publishDisp(float* disp) const;

    double sr = 48000.0;
    int nBands = 20;
    int nRunQuads = 5;
    bool boot = true;
    bool retro = false;
    float flo = 80.f;
    float fhi = 18000.f;
    float bwPct = 100.f;
    DesignKey key {};

    float formantTarget = 0.f;
    float formantSm = 0.f;
    float formantApplied = 1.0e9f;
    float nTrim = 1.f;

    int depthMode = 1;
    float depthK = 1.f;
    bool gateOff = true;
    float gateOpenTh = 0.f;
    float gateCloseTh = 0.f;
    bool uvOff = true;
    float uvLin = 0.f;
    float uvThr = 0.f;
    float uvEnergyLin = 0.f;
    float sens = 50.f;

    float cAtt = 0.f, cRel = 0.f, cEnhAtt = 0.f, cEnhRel = 0.f;
    float cEnhPkAtt = 0.f, cEnhPkRel = 0.f, cGate = 0.f, cDuck = 0.f;
    float cMacro = 0.f, cBmix = 0.f, cAttSm = 0.f, cUvAtt = 0.f, cUvRel = 0.f;
    float cGlide = 0.f;

    float lvl = 1.f, lvlTarget = 1.f;
    float wet = 1.f, wetTarget = 1.f;
    float dev = 1.f, devTarget = 1.f;
    float enhMix = 0.f, enhTarget = 0.f;
    float duck = 1.f, duckTarget = 1.f;
    float enhPeak = 0.f;
    float uvEnv = 0.f;

    int wantCarrier = 0;
    int wantCh = 1;
    int activeCarrier = -1;
    int activeCh = -1;
    bool pending = false;
    int pendingCarrier = 0;
    int pendingCh = 1;

    int gainLaw = 1;
    bool doEnhance = false;
    bool shareSynth = false;
    bool lrMode = false;

    float noiseRate = 1.f;
    float noiseCrackle = 1.f;
    float holdN = 1.f;
    float shPos = 0.f;
    float shL = 0.f;
    float shR = 0.f;
    uint32_t rng = 0xA3C59AC3u;

    int oscWave = 0;
    float ptTranspose = 1.f;
    float ptTarget = 100.f;
    float ptTargetLog = 4.605170186f;
    float ptLog = 4.605170186f;
    float oscPhase = 0.f;
    float tLow = 60.f;
    float tHigh = 1000.f;

    int decim = 6;
    float srDec = 8000.f;
    float lpA1 = 0, lpA2 = 0, lpA3 = 0, lpK = 1;
    float decA1 = 0, decA2 = 0, decB1 = 0, decB2 = 0;
    int decI = 0;
    int hopN = 32;
    int hopCount = 0;
    int ringW = 0;
    int ringFill = 0;
    float yinGate2 = 1.0e-5f;
    static constexpr int kRing = 4096;
    std::array<float, kRing> ring {};
    std::array<float, 4096> ybuf {};
    std::array<float, 2048> yinD {};

    std::array<Band, kVocMaxBands> bands {};
    std::array<Quad, kVocQuads> quads {};

    float subInL[kVocSub] {};
    float subInR[kVocSub] {};
    float subModL[kVocSub] {};
    float subModR[kVocSub] {};
    float subCarL[kVocSub] {};
    float subCarR[kVocSub] {};
    float subSumL[kVocSub] {};
    float subSumR[kVocSub] {};
    float enhArr[kVocSub] {};
    float duckArr[kVocSub] {};
    float wetArr[kVocSub] {};
    float devArr[kVocSub] {};
    float lvlArr[kVocSub] {};
    float eAll[kVocSub] {};
    float eHi[kVocSub] {};

    V4 envHistL[kVocQuads][kVocSub];
    V4 envHistR[kVocQuads][kVocSub];
    V4 wHistL[kVocQuads][kVocSub];
    V4 wHistR[kVocQuads][kVocSub];
    V4 gHistL[kVocQuads][kVocSub];
    V4 gHistR[kVocQuads][kVocSub];
    V4 bpHistL[kVocQuads][kVocSub];
    V4 bpHistR[kVocQuads][kVocSub];
    V4 rmsHistL[kVocQuads][kVocSub];
    V4 rmsHistR[kVocQuads][kVocSub];
};
