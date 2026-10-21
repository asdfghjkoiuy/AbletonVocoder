#pragma once

#include "VocoderEngine.h"

#include <juce_audio_processors/juce_audio_processors.h>

namespace voc
{
inline constexpr const char* carrier = "carrier";
inline constexpr const char* bands = "bands";
inline constexpr const char* rangeLow = "rangeLow";
inline constexpr const char* rangeHigh = "rangeHigh";
inline constexpr const char* bw = "bw";
inline constexpr const char* filterMode = "filterMode";
inline constexpr const char* gate = "gate";
inline constexpr const char* level = "level";
inline constexpr const char* depth = "depth";
inline constexpr const char* attack = "attack";
inline constexpr const char* release = "release";
inline constexpr const char* formant = "formant";
inline constexpr const char* dryWet = "dryWet";
inline constexpr const char* channels = "channels";
inline constexpr const char* enhance = "enhance";
inline constexpr const char* unvoiced = "unvoiced";
inline constexpr const char* sens = "sens";
inline constexpr const char* uvSpeed = "uvSpeed";
inline constexpr const char* noiseRate = "noiseRate";
inline constexpr const char* noiseCrackle = "noiseCrackle";
inline constexpr const char* oscWave = "oscWave";
inline constexpr const char* oscPitch = "oscPitch";
inline constexpr const char* trackLow = "trackLow";
inline constexpr const char* trackHigh = "trackHigh";
inline constexpr const char* deviceOn = "deviceOn";
} // namespace voc

class AbletonVocoderProcessor : public juce::AudioProcessor
{
public:
    AbletonVocoderProcessor();
    ~AbletonVocoderProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Ableton Vocoder"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 1.5; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;
    void reset() override;

    void setBandAttDb(int index, float db);
    float getBandAttDb(int index) const;
    void resetAllBands();
#if defined(VOC_ENABLE_TEST_HOOKS)
    void debugInjectCorruptState(float val) noexcept { engine.debugInjectCorruptState(val); }
#endif

    juce::AudioProcessorValueTreeState apvts;
    std::atomic<float> bandAttDb[kVocMaxBands];
    std::atomic<float> dispEnv[kVocMaxBands];
    std::atomic<uint32_t> stateRevision { 1 };

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    VocoderParams readParams() const noexcept;

    struct ParamPtrs
    {
        std::atomic<float>* carrier = nullptr;
        std::atomic<float>* bands = nullptr;
        std::atomic<float>* rangeLow = nullptr;
        std::atomic<float>* rangeHigh = nullptr;
        std::atomic<float>* bw = nullptr;
        std::atomic<float>* filterMode = nullptr;
        std::atomic<float>* gate = nullptr;
        std::atomic<float>* level = nullptr;
        std::atomic<float>* depth = nullptr;
        std::atomic<float>* attack = nullptr;
        std::atomic<float>* release = nullptr;
        std::atomic<float>* formant = nullptr;
        std::atomic<float>* dryWet = nullptr;
        std::atomic<float>* channels = nullptr;
        std::atomic<float>* enhance = nullptr;
        std::atomic<float>* unvoiced = nullptr;
        std::atomic<float>* sens = nullptr;
        std::atomic<float>* uvSpeed = nullptr;
        std::atomic<float>* noiseRate = nullptr;
        std::atomic<float>* noiseCrackle = nullptr;
        std::atomic<float>* oscWave = nullptr;
        std::atomic<float>* oscPitch = nullptr;
        std::atomic<float>* trackLow = nullptr;
        std::atomic<float>* trackHigh = nullptr;
        std::atomic<float>* deviceOn = nullptr;
    } paramPtrs;

    VocoderEngine engine;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AbletonVocoderProcessor)
};
