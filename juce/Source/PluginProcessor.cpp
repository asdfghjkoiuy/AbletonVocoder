#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <cmath>

namespace
{
juce::NormalisableRange<float> logRange(float lo, float hi)
{
    juce::NormalisableRange<float> range(lo, hi);
    range.setSkewForCentre(std::sqrt(lo * hi));
    return range;
}

inline float loadPtr(const std::atomic<float>* p) noexcept
{
    return p != nullptr ? p->load(std::memory_order_relaxed) : 0.f;
}

inline float loadFinite(const std::atomic<float>* p, float fallback) noexcept
{
    const float v = loadPtr(p);
    return vocIsFinite(v) ? v : fallback;
}
} // namespace

AbletonVocoderProcessor::AbletonVocoderProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Modulator", juce::AudioChannelSet::stereo(), true)
                         .withInput("Carrier", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "AbletonVocoder", createLayout())
{
    for (int i = 0; i < kVocMaxBands; ++i)
    {
        bandAttDb[i].store(0.f, std::memory_order_relaxed);
        dispEnv[i].store(0.f, std::memory_order_relaxed);
    }

    paramPtrs.carrier      = apvts.getRawParameterValue(voc::carrier);
    paramPtrs.bands        = apvts.getRawParameterValue(voc::bands);
    paramPtrs.rangeLow     = apvts.getRawParameterValue(voc::rangeLow);
    paramPtrs.rangeHigh    = apvts.getRawParameterValue(voc::rangeHigh);
    paramPtrs.bw           = apvts.getRawParameterValue(voc::bw);
    paramPtrs.filterMode   = apvts.getRawParameterValue(voc::filterMode);
    paramPtrs.gate         = apvts.getRawParameterValue(voc::gate);
    paramPtrs.level        = apvts.getRawParameterValue(voc::level);
    paramPtrs.depth        = apvts.getRawParameterValue(voc::depth);
    paramPtrs.attack       = apvts.getRawParameterValue(voc::attack);
    paramPtrs.release      = apvts.getRawParameterValue(voc::release);
    paramPtrs.formant      = apvts.getRawParameterValue(voc::formant);
    paramPtrs.dryWet       = apvts.getRawParameterValue(voc::dryWet);
    paramPtrs.channels     = apvts.getRawParameterValue(voc::channels);
    paramPtrs.enhance      = apvts.getRawParameterValue(voc::enhance);
    paramPtrs.unvoiced     = apvts.getRawParameterValue(voc::unvoiced);
    paramPtrs.sens         = apvts.getRawParameterValue(voc::sens);
    paramPtrs.uvSpeed      = apvts.getRawParameterValue(voc::uvSpeed);
    paramPtrs.noiseRate    = apvts.getRawParameterValue(voc::noiseRate);
    paramPtrs.noiseCrackle = apvts.getRawParameterValue(voc::noiseCrackle);
    paramPtrs.oscWave      = apvts.getRawParameterValue(voc::oscWave);
    paramPtrs.oscPitch     = apvts.getRawParameterValue(voc::oscPitch);
    paramPtrs.trackLow     = apvts.getRawParameterValue(voc::trackLow);
    paramPtrs.trackHigh    = apvts.getRawParameterValue(voc::trackHigh);
    paramPtrs.deviceOn     = apvts.getRawParameterValue(voc::deviceOn);
}

AbletonVocoderProcessor::~AbletonVocoderProcessor() = default;

juce::AudioProcessorValueTreeState::ParameterLayout AbletonVocoderProcessor::createLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { voc::carrier, 1 }, "Carrier",
        juce::StringArray { "Noise", "External", "Modulator", "Pitch Tracking" }, 0));
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { voc::bands, 1 }, "Bands",
        juce::StringArray { "4", "8", "12", "16", "20", "24", "28", "32", "36", "40" }, 4));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::rangeLow, 1 }, "Range Low (Hz)", logRange(20.f, 2000.f), 80.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::rangeHigh, 1 }, "Range High (Hz)", logRange(1000.f, 20000.f), 18000.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::bw, 1 }, "BW (%)", juce::NormalisableRange<float>(10.f, 200.f), 100.f));
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { voc::filterMode, 1 }, "Filter Mode",
        juce::StringArray { "Precise", "Retro" }, 0));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::gate, 1 }, "Gate (dB)", juce::NormalisableRange<float>(-96.f, 0.f), -96.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::level, 1 }, "Level (dB)", juce::NormalisableRange<float>(-24.f, 24.f), 0.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::depth, 1 }, "Depth (%)", juce::NormalisableRange<float>(0.f, 200.f), 100.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::attack, 1 }, "Attack (ms)", logRange(0.1f, 100.f), 5.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::release, 1 }, "Release (ms)", logRange(1.f, 1000.f), 80.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::formant, 1 }, "Formant (st)", juce::NormalisableRange<float>(-12.f, 12.f), 0.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::dryWet, 1 }, "Dry/Wet (%)", juce::NormalisableRange<float>(0.f, 100.f), 100.f));
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { voc::channels, 1 }, "Channels",
        juce::StringArray { "Mono", "Stereo", "L/R" }, 1));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { voc::enhance, 1 }, "Enhance", false));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::unvoiced, 1 }, "Unvoiced (dB)", juce::NormalisableRange<float>(-96.f, 0.f), -96.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::sens, 1 }, "Sens (%)", juce::NormalisableRange<float>(0.f, 100.f), 50.f));
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { voc::uvSpeed, 1 }, "Unvoiced Speed",
        juce::StringArray { "Slow", "Fast" }, 0));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::noiseRate, 1 }, "Noise Rate", juce::NormalisableRange<float>(0.f, 1.f), 1.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::noiseCrackle, 1 }, "Noise Crackle", juce::NormalisableRange<float>(0.f, 1.f), 1.f));
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { voc::oscWave, 1 }, "Osc Wave",
        juce::StringArray { "Saw", "Pulse 50%", "Pulse 25%", "Pulse 12.5%" }, 0));
    layout.add(std::make_unique<juce::AudioParameterInt>(
        juce::ParameterID { voc::oscPitch, 1 }, "Osc Pitch (st)", -24, 24, 0));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::trackLow, 1 }, "Track Low (Hz)", logRange(20.f, 1000.f), 60.f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { voc::trackHigh, 1 }, "Track High (Hz)", logRange(100.f, 2000.f), 1000.f));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { voc::deviceOn, 1 }, "Device On", true));
    return layout;
}

void AbletonVocoderProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    engine.prepare(sampleRate, samplesPerBlock);
}

void AbletonVocoderProcessor::releaseResources() {}

void AbletonVocoderProcessor::reset()
{
    engine.reset();
    for (int i = 0; i < kVocMaxBands; ++i)
        dispEnv[i].store(0.f, std::memory_order_relaxed);
}

bool AbletonVocoderProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto mainIn = layouts.getMainInputChannelSet();
    const auto mainOut = layouts.getMainOutputChannelSet();
    if (mainIn != juce::AudioChannelSet::stereo() && mainIn != juce::AudioChannelSet::mono())
        return false;
    if (mainOut != juce::AudioChannelSet::stereo() && mainOut != juce::AudioChannelSet::mono())
        return false;
    if (layouts.inputBuses.size() > 1)
    {
        const auto side = layouts.getChannelSet(true, 1);
        if (!side.isDisabled() && side != juce::AudioChannelSet::stereo() && side != juce::AudioChannelSet::mono())
            return false;
    }
    return true;
}

VocoderParams AbletonVocoderProcessor::readParams() const noexcept
{
    VocoderParams p;
    p.carrier      = (int) std::lround(loadFinite(paramPtrs.carrier, 0.f));
    p.bands        = ((int) std::lround(loadFinite(paramPtrs.bands, 4.f)) + 1) * 4;
    p.rangeLow     = loadFinite(paramPtrs.rangeLow, 80.f);
    p.rangeHigh    = loadFinite(paramPtrs.rangeHigh, 18000.f);
    p.bw           = loadFinite(paramPtrs.bw, 100.f);
    p.filterMode   = (int) std::lround(loadFinite(paramPtrs.filterMode, 0.f));
    p.gateDb       = loadFinite(paramPtrs.gate, -96.f);
    p.levelDb      = loadFinite(paramPtrs.level, 0.f);
    p.depth        = loadFinite(paramPtrs.depth, 100.f);
    p.attackMs     = loadFinite(paramPtrs.attack, 5.f);
    p.releaseMs    = loadFinite(paramPtrs.release, 80.f);
    p.formantSt    = loadFinite(paramPtrs.formant, 0.f);
    p.dryWet       = loadFinite(paramPtrs.dryWet, 100.f);
    p.channels     = (int) std::lround(loadFinite(paramPtrs.channels, 1.f));
    p.enhance      = loadFinite(paramPtrs.enhance, 0.f) > 0.5f;
    p.unvoicedDb   = loadFinite(paramPtrs.unvoiced, -96.f);
    p.sens         = loadFinite(paramPtrs.sens, 50.f);
    p.uvSpeed      = (int) std::lround(loadFinite(paramPtrs.uvSpeed, 0.f));
    p.noiseRate    = loadFinite(paramPtrs.noiseRate, 1.f);
    p.noiseCrackle = loadFinite(paramPtrs.noiseCrackle, 1.f);
    p.oscWave      = (int) std::lround(loadFinite(paramPtrs.oscWave, 0.f));
    p.oscPitch     = (int) std::lround(loadFinite(paramPtrs.oscPitch, 0.f));
    p.trackLow     = loadFinite(paramPtrs.trackLow, 60.f);
    p.trackHigh    = loadFinite(paramPtrs.trackHigh, 1000.f);
    p.deviceOn     = loadFinite(paramPtrs.deviceOn, 1.f) > 0.5f;
    return p;
}

void AbletonVocoderProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int n = buffer.getNumSamples();
    const int ch = buffer.getNumChannels();
    if (n <= 0 || ch < 1)
        return;

    auto mainIn = getBusBuffer(buffer, true, 0);
    auto mainOut = getBusBuffer(buffer, false, 0);
    const int inCh = mainIn.getNumChannels();
    const int outCh = mainOut.getNumChannels();
    if (inCh < 1 || outCh < 1)
        return;

    const float* modL = mainIn.getReadPointer(0);
    const float* modR = inCh >= 2 ? mainIn.getReadPointer(1) : modL;

    const float* extL = nullptr;
    const float* extR = nullptr;
    if (auto* sideBus = getBus(true, 1); sideBus != nullptr && sideBus->isEnabled() && getBusCount(true) > 1)
    {
        auto sideBuf = getBusBuffer(buffer, true, 1);
        if (sideBuf.getNumChannels() >= 1)
        {
            extL = sideBuf.getReadPointer(0);
            extR = sideBuf.getNumChannels() >= 2 ? sideBuf.getReadPointer(1) : extL;
        }
    }

    float att[kVocMaxBands];
    float disp[kVocMaxBands];
    for (int i = 0; i < kVocMaxBands; ++i)
        att[i] = bandAttDb[i].load(std::memory_order_relaxed);

    const VocoderParams p = readParams();
    float* outPtrL = mainOut.getWritePointer(0);
    float* outPtrR = outCh >= 2 ? mainOut.getWritePointer(1) : nullptr;
    const bool sideAliasesOutR = (outPtrR != nullptr && (outPtrR == extL || outPtrR == extR));

    if (outPtrR != nullptr && !sideAliasesOutR)
    {
        engine.process(modL, modR, extL, extR, outPtrL, outPtrR, n, p, att, disp);
    }
    else
    {
        float scratchL[kVocSub];
        float scratchR[kVocSub];
        int offset = 0;
        while (offset < n)
        {
            const int m = std::min(kVocSub, n - offset);
            engine.process(modL + offset, modR + offset,
                           extL != nullptr ? extL + offset : nullptr,
                           extR != nullptr ? extR + offset : nullptr,
                           scratchL, scratchR,
                           m, p, att, disp);
            std::memcpy(outPtrL + offset, scratchL, sizeof(float) * (size_t) m);
            if (outPtrR != nullptr)
                std::memcpy(outPtrR + offset, scratchR, sizeof(float) * (size_t) m);
            offset += m;
        }
    }

    for (int i = 0; i < kVocMaxBands; ++i)
        dispEnv[i].store(disp[i], std::memory_order_relaxed);
}

juce::AudioProcessorEditor* AbletonVocoderProcessor::createEditor()
{
    return new AbletonVocoderEditor(*this);
}

void AbletonVocoderProcessor::setBandAttDb(int index, float db)
{
    if (index < 0 || index >= kVocMaxBands)
        return;
    if (!vocIsFinite(db))
        db = 0.f;
    bandAttDb[index].store(juce::jlimit(-60.f, 0.f, db), std::memory_order_relaxed);
    stateRevision.fetch_add(1, std::memory_order_relaxed);
}

float AbletonVocoderProcessor::getBandAttDb(int index) const
{
    if (index < 0 || index >= kVocMaxBands)
        return 0.f;
    return bandAttDb[index].load(std::memory_order_relaxed);
}

void AbletonVocoderProcessor::resetAllBands()
{
    for (int i = 0; i < kVocMaxBands; ++i)
        bandAttDb[i].store(0.f, std::memory_order_relaxed);
    stateRevision.fetch_add(1, std::memory_order_relaxed);
}

void AbletonVocoderProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    juce::ValueTree state = apvts.copyState();
    auto bandsNode = state.getOrCreateChildWithName("bands", nullptr);
    for (int i = 0; i < kVocMaxBands; ++i)
        bandsNode.setProperty("a" + juce::String(i), bandAttDb[i].load(std::memory_order_relaxed), nullptr);
    if (auto xml = state.createXml())
        copyXmlToBinary(*xml, destData);
}

void AbletonVocoderProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    auto xml = getXmlFromBinary(data, sizeInBytes);
    if (xml == nullptr || !xml->hasTagName(apvts.state.getType().toString()))
        return;
    const auto tree = juce::ValueTree::fromXml(*xml);
    if (!tree.isValid())
        return;
    apvts.replaceState(tree);
    for (auto* param : getParameters())
    {
        if (param == nullptr)
            continue;
        const float norm = param->getValue();
        if (!vocIsFinite(norm))
            param->setValue(param->getDefaultValue());
    }
    auto bandsNode = tree.getChildWithName("bands");
    for (int i = 0; i < kVocMaxBands; ++i)
    {
        float db = 0.f;
        if (bandsNode.isValid())
            db = (float) bandsNode.getProperty("a" + juce::String(i), 0.0f);
        if (!vocIsFinite(db))
            db = 0.f;
        bandAttDb[i].store(juce::jlimit(-60.f, 0.f, db), std::memory_order_relaxed);
    }
    stateRevision.fetch_add(1, std::memory_order_relaxed);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AbletonVocoderProcessor();
}
