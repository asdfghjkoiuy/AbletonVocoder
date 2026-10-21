#include "PluginProcessor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace
{
bool nonFinite(float x)
{
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    return (u & 0x7f800000u) == 0x7f800000u;
}

float peakOf(const std::vector<float>& x)
{
    float p = 0.f;
    for (float v : x)
        p = std::max(p, std::abs(v));
    return p;
}

double rmsOf(const float* x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += (double) x[i] * (double) x[i];
    return n > 0 ? std::sqrt(s / (double) n) : 0.0;
}

void scaleRms(std::vector<float>& x, float db)
{
    double s = 0.0;
    for (float v : x)
        s += (double) v * (double) v;
    const double rms = std::sqrt(s / (double) std::max<size_t>(1, x.size()));
    const double target = std::exp((double) db * 2.302585092994046 / 20.0);
    if (rms > 1.0e-12)
    {
        const float g = (float) (target / rms);
        for (float& v : x)
            v *= g;
    }
}

std::vector<float> loadWav(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return {};
    auto read32 = [&]() {
        uint32_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 4);
        return v;
    };
    auto read16 = [&]() {
        uint16_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 2);
        return v;
    };
    if (read32() != 0x46464952)
        return {};
    read32();
    if (read32() != 0x45564157)
        return {};
    uint16_t format = 1, channels = 1, bits = 16;
    uint32_t rate = 48000;
    std::vector<float> out;
    while (in)
    {
        const uint32_t id = read32();
        const uint32_t size = read32();
        if (!in)
            break;
        if (id == 0x20746d66)
        {
            format = read16();
            channels = read16();
            rate = read32();
            read32();
            read16();
            bits = read16();
            if (size > 16)
                in.seekg(size - 16, std::ios::cur);
        }
        else if (id == 0x61746164)
        {
            const int frames = channels > 0 ? (int) (size / (channels * (bits / 8))) : 0;
            out.resize((size_t) frames);
            for (int i = 0; i < frames; ++i)
            {
                float acc = 0.f;
                for (int c = 0; c < channels; ++c)
                {
                    float s = 0.f;
                    if (format == 3 && bits == 32)
                        in.read(reinterpret_cast<char*>(&s), 4);
                    else if (format == 1 && bits == 16)
                    {
                        int16_t v = 0;
                        in.read(reinterpret_cast<char*>(&v), 2);
                        s = (float) v / 32768.f;
                    }
                    else
                    {
                        in.seekg(bits / 8, std::ios::cur);
                    }
                    if (c == 0)
                        acc = s;
                }
                out[(size_t) i] = acc;
            }
            juce::ignoreUnused(rate);
            break;
        }
        else
        {
            in.seekg(size + (size & 1u), std::ios::cur);
        }
    }
    return out;
}

std::vector<float> loadOrSynth(const std::string& file, int n, bool saw)
{
    for (const char* prefix : { "tests/signals/", "../tests/signals/", "../../tests/signals/" })
    {
        auto data = loadWav(std::string(prefix) + file);
        if ((int) data.size() >= n)
        {
            data.resize((size_t) n);
            return data;
        }
    }
    std::vector<float> data((size_t) n, 0.f);
    for (int i = 0; i < n; ++i)
    {
        const float t = (float) i / 48000.f;
        if (saw)
        {
            auto ph = [](float f, float tt) {
                float p = std::fmod(f * tt, 1.f);
                return 2.f * p - 1.f;
            };
            data[(size_t) i] = 0.5f * ph(110.f, t) + 0.35f * ph(130.81f, t) + 0.25f * ph(164.81f, t);
        }
        else
        {
            const float env = 0.5f + 0.5f * std::sin(2.f * 3.14159265f * 4.f * t);
            float h = 0.f;
            for (int k = 1; k <= 12; ++k)
                h += std::sin(2.f * 3.14159265f * (90.f + 40.f * std::sin(2.f * t)) * (float) k * t) / (float) k;
            data[(size_t) i] = env * h * 0.2f;
        }
    }
    return data;
}

void setParam(AbletonVocoderProcessor& proc, const char* id, float value)
{
    if (auto* p = dynamic_cast<juce::RangedAudioParameter*>(proc.apvts.getParameter(id)))
    {
        p->beginChangeGesture();
        p->setValueNotifyingHost(p->convertTo0to1(value));
        p->endChangeGesture();
    }
}

void prepare(AbletonVocoderProcessor& proc)
{
    juce::AudioProcessor::BusesLayout layout;
    layout.inputBuses.add(juce::AudioChannelSet::stereo());
    layout.inputBuses.add(juce::AudioChannelSet::stereo());
    layout.outputBuses.add(juce::AudioChannelSet::stereo());
    proc.setBusesLayout(layout);
    proc.setRateAndBufferSizeDetails(48000.0, 256);
    proc.prepareToPlay(48000.0, 256);
}

struct Stats
{
    double rtf = 0;
    double wall = 0;
    bool finite = true;
    float peak = 0;
};

Stats render(AbletonVocoderProcessor& proc, const std::vector<float>& mod, const std::vector<float>& car)
{
    constexpr int kBlock = 256;
    constexpr double kSr = 48000.0;
    const int n = (int) std::min(mod.size(), car.size());
    juce::AudioBuffer<float> buffer(4, kBlock);
    juce::MidiBuffer midi;
    std::vector<float> out;
    out.reserve((size_t) n * 2u);

    buffer.clear();
    proc.processBlock(buffer, midi);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i += kBlock)
    {
        const int m = std::min(kBlock, n - i);
        buffer.setSize(4, m, false, false, true);
        for (int s = 0; s < m; ++s)
        {
            buffer.setSample(0, s, mod[(size_t) (i + s)]);
            buffer.setSample(1, s, mod[(size_t) (i + s)]);
            buffer.setSample(2, s, car[(size_t) (i + s)]);
            buffer.setSample(3, s, car[(size_t) (i + s)]);
        }
        proc.processBlock(buffer, midi);
        for (int s = 0; s < m; ++s)
        {
            out.push_back(buffer.getSample(0, s));
            out.push_back(buffer.getSample(1, s));
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    Stats st;
    st.wall = wall;
    st.rtf = wall > 0.0 ? ((double) n / kSr) / wall : 0.0;
    st.peak = peakOf(out);
    for (float v : out)
        if (nonFinite(v))
            st.finite = false;
    return st;
}

double median3(double a, double b, double c)
{
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return b;
}
} // namespace

int main()
{
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    juce::ScopedJuceInitialiser_GUI juceInit;
    constexpr int kN = 48000 * 4;
    constexpr double kSr = 48000.0;

    auto speech = loadOrSynth("speech_trimmed_48000.wav", kN, false);
    auto chord = loadOrSynth("saw_minor_48000.wav", kN, true);
    auto voice = loadOrSynth("synth_voice_48000.wav", kN, false);
    if ((int) speech.size() > kN) speech.resize(kN);
    if ((int) chord.size() > kN) chord.resize(kN);
    if ((int) voice.size() > kN) voice.resize(kN);
    if ((int) speech.size() < kN || (int) chord.size() < kN || (int) voice.size() < kN)
    {
        std::cerr << "signal shorter than 4 seconds\n";
        return 1;
    }
    scaleRms(speech, -18.f);

    auto bench = [&](const char* name, int carrier, int bandIndex, int channels, bool enhance, float unvoiced,
                     const std::vector<float>& mod, const std::vector<float>& car) {
        AbletonVocoderProcessor proc;
        setParam(proc, voc::carrier, (float) carrier);
        setParam(proc, voc::bands, (float) bandIndex);
        setParam(proc, voc::channels, (float) channels);
        setParam(proc, voc::enhance, enhance ? 1.f : 0.f);
        setParam(proc, voc::unvoiced, unvoiced);
        setParam(proc, voc::dryWet, 100.f);
        setParam(proc, voc::depth, 100.f);
        setParam(proc, voc::deviceOn, 1.f);
        setParam(proc, voc::oscWave, 0.f);
        setParam(proc, voc::trackLow, 60.f);
        setParam(proc, voc::trackHigh, 1000.f);
        prepare(proc);
        Stats runs[3];
        for (int i = 0; i < 3; ++i)
        {
            proc.reset();
            runs[i] = render(proc, mod, car);
        }
        const double rtf = median3(runs[0].rtf, runs[1].rtf, runs[2].rtf);
        const bool finite = runs[0].finite && runs[1].finite && runs[2].finite;
        const float peak = std::max(runs[0].peak, std::max(runs[1].peak, runs[2].peak));
        std::cout << name << "  realtime " << rtf << " x"
                  << "   core " << (100.0 / rtf) << " %"
                  << "   wall " << ((double) kN / kSr) / rtf << " s"
                  << "   peak " << peak
                  << "   finite " << (finite ? "yes" : "NO")
                  << "\n";
        std::cout << "  runs";
        for (const auto& r : runs)
            std::cout << " " << r.rtf;
        std::cout << "\n";
        return std::pair<double, bool>(rtf, finite && peak > 1.0e-4f);
    };

    std::cout << "Ableton Vocoder JUCE benchmark  48 kHz  4.00 s  block 256\n";
    std::cout << "JSFX reference: T11a ~10.3x   T11b ~2.3x\n";
    const auto a = bench("T11a  20 bands Stereo External", 1, 4, 1, false, -96.f, speech, chord);
    std::vector<float> zeros((size_t) kN, 0.f);
    const auto b = bench("T11b  40 bands L/R Pitch+Enhance+Unvoiced", 3, 9, 2, true, 0.f, voice, zeros);

    AbletonVocoderProcessor silenceProc;
    setParam(silenceProc, voc::carrier, 1.f);
    setParam(silenceProc, voc::bands, 4.f);
    setParam(silenceProc, voc::channels, 1.f);
    setParam(silenceProc, voc::dryWet, 100.f);
    setParam(silenceProc, voc::depth, 100.f);
    setParam(silenceProc, voc::enhance, 0.f);
    setParam(silenceProc, voc::unvoiced, -96.f);
    prepare(silenceProc);
    juce::AudioBuffer<float> buf(4, 256);
    juce::MidiBuffer midi;
    float silencePeak = 0.f;
    bool silenceFinite = true;
    for (int i = 0; i < 48000; i += 256)
    {
        buf.clear();
        silenceProc.processBlock(buf, midi);
        for (int s = 0; s < buf.getNumSamples(); ++s)
        {
            silencePeak = std::max(silencePeak, std::abs(buf.getSample(0, s)));
            silencePeak = std::max(silencePeak, std::abs(buf.getSample(1, s)));
            if (nonFinite(buf.getSample(0, s)) || nonFinite(buf.getSample(1, s)))
                silenceFinite = false;
        }
    }

    AbletonVocoderProcessor routeProc;
    setParam(routeProc, voc::carrier, 1.f);
    setParam(routeProc, voc::bands, 4.f);
    setParam(routeProc, voc::channels, 1.f);
    setParam(routeProc, voc::dryWet, 100.f);
    setParam(routeProc, voc::depth, 100.f);
    setParam(routeProc, voc::enhance, 0.f);
    setParam(routeProc, voc::unvoiced, -96.f);
    setParam(routeProc, voc::deviceOn, 1.f);
    prepare(routeProc);
    std::vector<float> routeOut;
    std::vector<float> routeDry;
    const int routeN = 48000;
    const int switchAt = 24064;
    for (int i = 0; i < routeN; i += 256)
    {
        if (i == switchAt)
            setParam(routeProc, voc::carrier, 0.f);
        buf.setSize(4, 256, false, false, true);
        for (int s = 0; s < 256; ++s)
        {
            const float t = (float) (i + s) / 48000.f;
            const float m = 0.25f * std::sin(2.f * 3.14159265f * 440.f * t);
            const uint32_t u = (uint32_t) (i + s) * 1103515245u + 12345u;
            const float c = (float) (u >> 8) * (1.f / 8388608.f) - 1.f;
            buf.setSample(0, s, m);
            buf.setSample(1, s, m);
            buf.setSample(2, s, c * 0.25f);
            buf.setSample(3, s, c * 0.2f);
            routeDry.push_back(m);
        }
        routeProc.processBlock(buf, midi);
        for (int s = 0; s < 256; ++s)
            routeOut.push_back(0.5f * (buf.getSample(0, s) + buf.getSample(1, s)));
    }
    const int duck0 = switchAt + (int) (0.003 * 48000);
    const int duck1 = switchAt + (int) (0.007 * 48000);
    const double duckRms = rmsOf(routeOut.data() + duck0, duck1 - duck0);
    const double dryRms = rmsOf(routeDry.data() + duck0, duck1 - duck0);
    const double preRms = rmsOf(routeOut.data() + switchAt - 4000, 4000);
    const double afterRms = rmsOf(routeOut.data() + switchAt + 8000, 4000);
    const bool noDryBleed = preRms > 0.004 && duckRms < 0.15 * dryRms && duckRms < 0.5 * preRms;

    AbletonVocoderProcessor eqProc;
    setParam(eqProc, voc::carrier, 1.f);
    setParam(eqProc, voc::bands, 4.f);
    setParam(eqProc, voc::channels, 1.f);
    setParam(eqProc, voc::dryWet, 100.f);
    setParam(eqProc, voc::depth, 100.f);
    setParam(eqProc, voc::enhance, 0.f);
    setParam(eqProc, voc::unvoiced, -96.f);
    prepare(eqProc);
    std::vector<float> rightOut;
    const int eqN = 36000;
    const int glitchAt = 24000;
    for (int i = 0; i < eqN; i += 256)
    {
        buf.setSize(4, 256, false, false, true);
        for (int s = 0; s < 256; ++s)
        {
            const uint32_t r0 = (uint32_t) (i + s) * 1664525u + 1013904223u;
            const uint32_t r1 = r0 * 22695477u + 1u;
            const float m = ((r0 >> 8) * (1.f / 16777216.f)) * 0.5f - 0.25f;
            const float cL = ((r1 >> 8) * (1.f / 16777216.f)) * 0.5f - 0.25f;
            float cR = 0.f;
            if (i + s == glitchAt)
                cR = cL;
            buf.setSample(0, s, m);
            buf.setSample(1, s, m * 0.8f);
            buf.setSample(2, s, cL);
            buf.setSample(3, s, cR);
        }
        eqProc.processBlock(buf, midi);
        for (int s = 0; s < 256; ++s)
            rightOut.push_back(buf.getSample(1, s));
    }
    const int ring0 = glitchAt + 96;
    const int ring1 = glitchAt + 96 + (int) (0.03 * 48000);
    const double rightRms = rmsOf(rightOut.data() + ring0, ring1 - ring0);
    const bool statesHeld = rightRms < 1.0e-3;

    // Test 5: NaN / Inf injection & recovery test
    uint32_t nanBits = 0x7fc00001u;
    float nanVal = 0.f;
    std::memcpy(&nanVal, &nanBits, 4);
    const bool sanityCheckWorks = nonFinite(nanVal);

    AbletonVocoderProcessor nanProc;
    setParam(nanProc, voc::carrier, 1.f);
    setParam(nanProc, voc::bands, 9.f);
    setParam(nanProc, voc::channels, 2.f);
    setParam(nanProc, voc::enhance, 1.f);
    setParam(nanProc, voc::unvoiced, 0.f);
    prepare(nanProc);
    bool nanCleanOutput = true;
    std::vector<float> postNanOut;
    for (int blk = 0; blk < 60; ++blk)
    {
        buf.setSize(4, 256, false, false, true);
        for (int s = 0; s < 256; ++s)
        {
            const int idx = (blk * 256 + s) % kN;
            buf.setSample(0, s, speech[(size_t) idx]);
            buf.setSample(1, s, speech[(size_t) idx]);
            buf.setSample(2, s, chord[(size_t) idx]);
            buf.setSample(3, s, chord[(size_t) idx]);
        }
        if (blk == 10)
        {
            buf.setSample(0, 17, nanVal);
            buf.setSample(2, 42, -nanVal);
            nanProc.debugInjectCorruptState(nanVal);
        }
        nanProc.processBlock(buf, midi);
        for (int s = 0; s < 256; ++s)
        {
            const float oL = buf.getSample(0, s);
            const float oR = buf.getSample(1, s);
            if (nonFinite(oL) || nonFinite(oR))
                nanCleanOutput = false;
            if (blk >= 20)
                postNanOut.push_back(oL);
        }
    }
    const double postNanRms = rmsOf(postNanOut.data(), (int) postNanOut.size());
    const bool nanRecovered = sanityCheckWorks && nanCleanOutput && postNanRms > 1.0e-3;

    // Test 6: Wake from sleep when Depth -> 0% on Noise carrier
    setParam(silenceProc, voc::carrier, 0.f);
    setParam(silenceProc, voc::depth, 100.f);
    for (int i = 0; i < 6400; i += 256)
    {
        buf.clear();
        silenceProc.processBlock(buf, midi);
    }
    setParam(silenceProc, voc::depth, 0.f);
    std::vector<float> wakeOut;
    for (int i = 0; i < 16384; i += 256)
    {
        buf.clear();
        silenceProc.processBlock(buf, midi);
        if (i >= 10240)
            for (int s = 0; s < 256; ++s)
                wakeOut.push_back(buf.getSample(0, s));
    }
    const double wakeRms = rmsOf(wakeOut.data(), (int) wakeOut.size());
    setParam(silenceProc, voc::depth, 100.f);
    for (int i = 0; i < 16384; i += 256)
    {
        buf.clear();
        silenceProc.processBlock(buf, midi);
    }
    setParam(silenceProc, voc::depth, 10.f);
    std::vector<float> wake10Out;
    for (int i = 0; i < 16384; i += 256)
    {
        buf.clear();
        silenceProc.processBlock(buf, midi);
        if (i >= 10240)
            for (int s = 0; s < 256; ++s)
                wake10Out.push_back(buf.getSample(0, s));
    }
    const double wake10Rms = rmsOf(wake10Out.data(), (int) wake10Out.size());
    const bool wakePass = wakeRms > 0.05 && wake10Rms > 0.02;

    // Test 7: Device Off -> On clears stale envelope tail
    AbletonVocoderProcessor devProc;
    setParam(devProc, voc::carrier, 0.f);
    setParam(devProc, voc::release, 800.f);
    setParam(devProc, voc::deviceOn, 1.f);
    prepare(devProc);
    for (int blk = 0; blk < 15; ++blk)
    {
        buf.setSize(4, 256, false, false, true);
        for (int s = 0; s < 256; ++s)
        {
            buf.setSample(0, s, speech[(size_t) (blk * 256 + s)]);
            buf.setSample(1, s, speech[(size_t) (blk * 256 + s)]);
            buf.setSample(2, s, 0.f);
            buf.setSample(3, s, 0.f);
        }
        devProc.processBlock(buf, midi);
    }
    setParam(devProc, voc::deviceOn, 0.f);
    for (int blk = 0; blk < 30; ++blk)
    {
        buf.clear();
        devProc.processBlock(buf, midi);
    }
    setParam(devProc, voc::deviceOn, 1.f);
    std::vector<float> reenableOut;
    for (int blk = 0; blk < 15; ++blk)
    {
        buf.clear();
        devProc.processBlock(buf, midi);
        for (int s = 0; s < 256; ++s)
            reenableOut.push_back(buf.getSample(0, s));
    }
    const double reenableRms = rmsOf(reenableOut.data(), (int) reenableOut.size());
    const bool devTailCleared = reenableRms < 1.0e-5;

    // Test 8: Smooth Depth 100% -> 0% transition (no step jump at 0% boundary)
    AbletonVocoderProcessor depthSweepProc;
    setParam(depthSweepProc, voc::carrier, 1.f);
    setParam(depthSweepProc, voc::bands, 4.f);
    setParam(depthSweepProc, voc::depth, 100.f);
    setParam(depthSweepProc, voc::dryWet, 100.f);
    prepare(depthSweepProc);
    double prevBlkPeak = 0.0;
    double maxBlkJump = 0.0;
    for (int blk = 0; blk < 40; ++blk)
    {
        if (blk == 5)
            setParam(depthSweepProc, voc::depth, 0.f);
        buf.setSize(4, 256, false, false, true);
        for (int s = 0; s < 256; ++s)
        {
            const float c = 0.25f * std::sin((float) (2.0 * 3.14159265358979323846 * 440.0 * (blk * 256 + s) / 48000.0));
            buf.setSample(0, s, 0.f);
            buf.setSample(1, s, 0.f);
            buf.setSample(2, s, c);
            buf.setSample(3, s, c);
        }
        depthSweepProc.processBlock(buf, midi);
        double blkPeak = 0.0;
        for (int s = 0; s < 256; ++s)
            blkPeak = std::max(blkPeak, (double) std::abs(buf.getSample(0, s)));
        if (blk >= 6)
            maxBlkJump = std::max(maxBlkJump, std::abs(blkPeak - prevBlkPeak));
        prevBlkPeak = blkPeak;
    }
    const bool depthSweepSmooth = maxBlkJump < 0.04 && prevBlkPeak > 0.15;

    // Test 9: H1 (no frozen envelope resurrection on Bands 40 -> 4 -> silence -> 40) & M2 (reset() clears tail and stays active)
    AbletonVocoderProcessor bandSwitchProc;
    setParam(bandSwitchProc, voc::carrier, 0.f);
    setParam(bandSwitchProc, voc::bands, 9.f); // 40 bands
    setParam(bandSwitchProc, voc::release, 1000.f);
    prepare(bandSwitchProc);
    for (int blk = 0; blk < 20; ++blk)
    {
        buf.setSize(4, 256, false, false, true);
        for (int s = 0; s < 256; ++s)
        {
            const int idx = (blk * 256 + s) % kN;
            buf.setSample(0, s, speech[(size_t) idx]);
            buf.setSample(1, s, speech[(size_t) idx]);
            buf.setSample(2, s, 0.f);
            buf.setSample(3, s, 0.f);
        }
        bandSwitchProc.processBlock(buf, midi);
    }
    setParam(bandSwitchProc, voc::bands, 0.f); // 4 bands
    double preSwitchPeak = 0.0;
    for (int blk = 0; blk < 60; ++blk)
    {
        buf.clear();
        bandSwitchProc.processBlock(buf, midi);
        if (blk == 59)
            for (int s = 0; s < 256; ++s)
                preSwitchPeak = std::max(preSwitchPeak, (double) std::abs(buf.getSample(0, s)));
    }
    setParam(bandSwitchProc, voc::bands, 9.f); // back to 40 bands while silent
    double frozenBandPeak = 0.0;
    for (int blk = 0; blk < 20; ++blk)
    {
        buf.clear();
        bandSwitchProc.processBlock(buf, midi);
        for (int s = 0; s < 256; ++s)
            frozenBandPeak = std::max(frozenBandPeak, (double) std::abs(buf.getSample(0, s)));
    }
    // Now excite again, call reset(), verify silence on silent block and normal output on active block
    for (int blk = 0; blk < 10; ++blk)
    {
        for (int s = 0; s < 256; ++s)
        {
            buf.setSample(0, s, speech[(size_t) (4096 + blk * 256 + s)]);
            buf.setSample(1, s, speech[(size_t) (4096 + blk * 256 + s)]);
        }
        bandSwitchProc.processBlock(buf, midi);
    }
    bandSwitchProc.reset();
    buf.clear();
    bandSwitchProc.processBlock(buf, midi);
    const double resetSilRms = rmsOf(buf.getReadPointer(0), 256);
    for (int blk = 0; blk < 4; ++blk)
    {
        for (int s = 0; s < 256; ++s)
        {
            buf.setSample(0, s, speech[(size_t) (4096 + blk * 256 + s)]);
            buf.setSample(1, s, speech[(size_t) (4096 + blk * 256 + s)]);
        }
        bandSwitchProc.processBlock(buf, midi);
    }
    const double postResetRms = rmsOf(buf.getReadPointer(0), 256);
    const bool bandAndResetPass = frozenBandPeak < 5.0e-4
                               && resetSilRms < 1.0e-6 && postResetRms > 1.0e-3;

    std::cout << "silence-in -> silence-out  peak " << silencePeak
              << "  finite " << (silenceFinite ? "yes" : "NO")
              << "  pass " << (silenceFinite && silencePeak < 1.0e-5f ? "yes" : "NO") << "\n";
    std::cout << "route duck (no dry bleed)  pre " << preRms
              << "  duck " << duckRms
              << "  dry " << dryRms
              << "  after " << afterRms
              << "  pass " << (noDryBleed ? "yes" : "NO") << "\n";
    std::cout << "stereo state held across equal sample  right-rms " << rightRms
              << "  pass " << (statesHeld ? "yes" : "NO") << "\n";
    std::cout << "nan injection & recovery  detector " << (sanityCheckWorks ? "ok" : "BROKEN")
              << "  clean " << (nanCleanOutput ? "yes" : "NO")
              << "  post-rms " << postNanRms
              << "  pass " << (nanRecovered ? "yes" : "NO") << "\n";
    std::cout << "wake on depth=0/10 & device-off clear  wake-rms " << wakeRms
              << "  wake10-rms " << wake10Rms
              << "  reenable-rms " << reenableRms
              << "  pass " << (wakePass && devTailCleared ? "yes" : "NO") << "\n";
    std::cout << "depth 100->0 smooth transition  max-blk-jump " << maxBlkJump
              << "  final-peak " << prevBlkPeak
              << "  pass " << (depthSweepSmooth ? "yes" : "NO") << "\n";
    std::cout << "frozen band clear & reset()  frozen-peak " << frozenBandPeak
              << "  reset-sil " << resetSilRms
              << "  post-reset " << postResetRms
              << "  pass " << (bandAndResetPass ? "yes" : "NO") << "\n";

    const bool ok = a.second && b.second && silenceFinite && silencePeak < 1.0e-5f
                 && noDryBleed && statesHeld && nanRecovered && wakePass && devTailCleared
                 && depthSweepSmooth && bandAndResetPass;
    std::cout << (ok ? "VERIFICATION PASS\n" : "VERIFICATION FAIL\n");
    return ok ? 0 : 1;
}
