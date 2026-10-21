#pragma once

#include "PluginProcessor.h"

class AbletonVocoderEditor : public juce::AudioProcessorEditor,
                             private juce::Timer
{
public:
    explicit AbletonVocoderEditor(AbletonVocoderProcessor&);
    ~AbletonVocoderEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override {}

    void mouseDown(const juce::MouseEvent&) override;
    void mouseDrag(const juce::MouseEvent&) override;
    void mouseUp(const juce::MouseEvent&) override;
    void mouseMove(const juce::MouseEvent&) override;
    void mouseExit(const juce::MouseEvent&) override;
    void mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails&) override;
    bool keyPressed(const juce::KeyPress&) override;
    void focusLost(FocusChangeType) override;

private:
    void timerCallback() override;
    float param(const char* id) const;
    void setParam(const char* id, float value);
    void beginParam(const char* id);
    void endParam(const char* id);
    juce::Rectangle<float> getMenuRect() const;
    int getMenuItemAt(juce::Point<float> p) const;

    AbletonVocoderProcessor& proc;

    enum class Drag { none, value, xy, band };
    Drag drag = Drag::none;
    const char* dragId = nullptr;
    float dragLo = 0, dragHi = 1, dragStep = 0, dragStart = 0, dragY = 0;
    bool dragLog = false;
    bool dragMoved = false;
    bool xyRate = false;
    bool xyCrack = false;

    int menu = 0;
    bool menuArmed = false;
    int menuHover = -1;
    juce::Point<float> menuDownPos {};
    uint32_t menuDownMs = 0;

    int bandLast = -1;
    float bandLastAtt = 0;
    bool bandDirty = false;

    uint32_t lastStateRev = 0;
    float lastParamSnap[25] {};
    float lastDispDb[kVocMaxBands] {};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AbletonVocoderEditor)
};
