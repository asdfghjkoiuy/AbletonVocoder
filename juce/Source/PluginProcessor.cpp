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

float rawOf(const juce::AudioProcessorValueTreeState& tree, const char* id)
{
    if (auto* v = tree.getRawParameterValue(id))
        return v->load();
    return 0.f;
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
        bandAttDb[i].store(0.f);
        dispEnv[i].store(0.f);
    }
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

bool AbletonVocoderProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    if (layouts.inputBuses.size() > 1)
    {
        const auto side = layouts.getChannelSet(true, 1);
        if (!side.isDisabled() && side != juce::AudioChannelSet::stereo())
            return false;
    }
    return true;
}

VocoderParams AbletonVocoderProcessor::readParams() const
{
    VocoderParams p;
    p.carrier = (int) std::lround(rawOf(apvts, voc::carrier));
    p.bands = ((int) std::lround(rawOf(apvts, voc::bands)) + 1) * 4;
    p.rangeLow = rawOf(apvts, voc::rangeLow);
    p.rangeHigh = rawOf(apvts, voc::rangeHigh);
    p.bw = rawOf(apvts, voc::bw);
    p.filterMode = (int) std::lround(rawOf(apvts, voc::filterMode));
    p.gateDb = rawOf(apvts, voc::gate);
    p.levelDb = rawOf(apvts, voc::level);
    p.depth = rawOf(apvts, voc::depth);
    p.attackMs = rawOf(apvts, voc::attack);
    p.releaseMs = rawOf(apvts, voc::release);
    p.formantSt = rawOf(apvts, voc::formant);
    p.dryWet = rawOf(apvts, voc::dryWet);
    p.channels = (int) std::lround(rawOf(apvts, voc::channels));
    p.enhance = rawOf(apvts, voc::enhance) > 0.5f;
    p.unvoicedDb = rawOf(apvts, voc::unvoiced);
    p.sens = rawOf(apvts, voc::sens);
    p.uvSpeed = (int) std::lround(rawOf(apvts, voc::uvSpeed));
    p.noiseRate = rawOf(apvts, voc::noiseRate);
    p.noiseCrackle = rawOf(apvts, voc::noiseCrackle);
    p.oscWave = (int) std::lround(rawOf(apvts, voc::oscWave));
    p.oscPitch = (int) std::lround(rawOf(apvts, voc::oscPitch));
    p.trackLow = rawOf(apvts, voc::trackLow);
    p.trackHigh = rawOf(apvts, voc::trackHigh);
    p.deviceOn = rawOf(apvts, voc::deviceOn) > 0.5f;
    return p;
}

void AbletonVocoderProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int n = buffer.getNumSamples();
    const int ch = buffer.getNumChannels();
    if (n <= 0 || ch < 2)
        return;

    auto* side = getBus(true, 1);
    const bool sideOn = side != nullptr && side->isEnabled() && ch >= 4;
    const float* extL = sideOn ? buffer.getReadPointer(2) : nullptr;
    const float* extR = sideOn ? buffer.getReadPointer(3) : nullptr;

    float att[kVocMaxBands];
    float disp[kVocMaxBands];
    for (int i = 0; i < kVocMaxBands; ++i)
        att[i] = bandAttDb[i].load(std::memory_order_relaxed);

    engine.process(buffer.getReadPointer(0), buffer.getReadPointer(1),
                   extL, extR,
                   buffer.getWritePointer(0), buffer.getWritePointer(1),
                   n, readParams(), att, disp);

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
    bandAttDb[index].store(juce::jlimit(-60.f, 0.f, db), std::memory_order_relaxed);
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
}

void AbletonVocoderProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    juce::ValueTree state = apvts.copyState();
    auto bandsNode = state.getOrCreateChildWithName("bands", nullptr);
    for (int i = 0; i < kVocMaxBands; ++i)
        bandsNode.setProperty("a" + juce::String(i), bandAttDb[i].load(), nullptr);
    if (auto xml = state.createXml())
        copyXmlToBinary(*xml, destData);
}

void AbletonVocoderProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary(data, sizeInBytes))
    {
        const auto tree = juce::ValueTree::fromXml(*xml);
        if (!tree.isValid())
            return;
        apvts.replaceState(tree);
        auto bandsNode = tree.getChildWithName("bands");
        for (int i = 0; i < kVocMaxBands; ++i)
        {
            const float db = bandsNode.isValid()
                ? (float) bandsNode.getProperty("a" + juce::String(i), 0.0f)
                : 0.f;
            bandAttDb[i].store(juce::jlimit(-60.f, 0.f, db));
        }
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AbletonVocoderProcessor();
}
