#include "PluginEditor.h"

#include <cmath>
#include <cstring>

namespace
{
const juce::Colour kOut(28, 29, 31);
const juce::Colour kBg(49, 51, 54);
const juce::Colour kPanel(32, 34, 36);
const juce::Colour kScreen(16, 18, 20);
const juce::Colour kText(230, 230, 230);
const juce::Colour kDim(148, 150, 154);
const juce::Colour kTeal(42, 210, 216);
const juce::Colour kOrange(255, 149, 18);
const juce::Colour kInk(36, 24, 14);
const juce::Colour kLine(18, 18, 20);
const juce::Colour kKnobC(22, 24, 26);
const juce::Colour kTrack(78, 80, 84);

constexpr const char* kAllParamIds[25] = {
    voc::carrier, voc::bands, voc::rangeLow, voc::rangeHigh, voc::bw,
    voc::filterMode, voc::gate, voc::level, voc::depth, voc::attack,
    voc::release, voc::formant, voc::dryWet, voc::channels, voc::enhance,
    voc::unvoiced, voc::sens, voc::uvSpeed, voc::noiseRate, voc::noiseCrackle,
    voc::oscWave, voc::oscPitch, voc::trackLow, voc::trackHigh, voc::deviceOn
};

struct Geo
{
    float s = 1.f, ox = 0.f, oy = 0.f, fs = 11.f, fb = 16.f, fm = 10.f;
    juce::Rectangle<float> carrier, ctx, pad, wave, pitch, low, high;
    juce::Rectangle<float> enhance, sens, speed, disp, bars;
    juce::Rectangle<float> bands, rLow, rHigh, bw, mode, gate, level;
    juce::Rectangle<float> chMono, chSt, chLR;
    juce::Point<float> power, uv, depth, attack, formant, release, dry;
    float powerR = 6.f, uvR = 15.f, depthR = 16.f, smallR = 14.f;

    float X(float v) const { return ox + v * s; }
    float Y(float v) const { return oy + v * s; }
    float S(float v) const { return v * s; }
    juce::Rectangle<float> box(float x, float y, float w, float h) const
    {
        return { X(x), Y(y), S(w), S(h) };
    }
    juce::Rectangle<float> around(juce::Point<float> c, float r) const
    {
        return { c.x - r - S(10.f), c.y - r - S(18.f), r * 2.f + S(20.f), r * 2.f + S(40.f) };
    }
};

Geo makeGeo(float w, float h)
{
    Geo g;
    g.s = std::min(w / 780.f, h / 290.f);
    if (g.s < 0.6f)
        g.s = 0.6f;
    g.ox = (w - 780.f * g.s) * 0.5f;
    g.oy = (h - 290.f * g.s) * 0.5f;
    g.fs = std::clamp(std::round(11.f * g.s), 8.f, 64.f);
    g.fb = std::clamp(std::round(16.f * g.s), 8.f, 72.f);
    g.fm = std::clamp(std::round(10.f * g.s), 8.f, 48.f);
    g.power = { g.X(16.f), g.Y(13.f) };
    g.powerR = g.S(6.5f);
    g.carrier = g.box(10, 42, 142, 18);
    g.ctx = g.box(10, 66, 142, 108);
    g.pad = g.box(16, 72, 130, 96);
    g.wave = g.box(16, 80, 130, 16);
    g.pitch = g.box(16, 102, 130, 16);
    g.low = g.box(16, 122, 130, 16);
    g.high = g.box(16, 142, 130, 16);
    g.enhance = g.box(10, 180, 68, 16);
    g.uv = { g.X(40.f), g.Y(232.f) };
    g.uvR = g.S(15.f);
    g.sens = g.box(90, 208, 58, 16);
    g.speed = g.box(90, 232, 58, 16);
    g.disp = g.box(164, 32, 432, 170);
    g.bars = g.disp.reduced(g.S(8.f));
    g.bands = g.box(164, 220, 48, 18);
    g.rLow = g.box(222, 220, 66, 16);
    g.rHigh = g.box(222, 240, 66, 16);
    g.bw = g.box(298, 220, 50, 18);
    g.mode = g.box(356, 220, 72, 18);
    g.gate = g.box(436, 220, 56, 18);
    g.level = g.box(500, 220, 62, 18);
    g.depth = { g.X(652.f), g.Y(70.f) };
    g.depthR = g.S(16.f);
    g.chMono = g.box(704, 40, 64, 16);
    g.chSt = g.box(704, 60, 64, 16);
    g.chLR = g.box(704, 80, 64, 16);
    g.attack = { g.X(652.f), g.Y(162.f) };
    g.formant = { g.X(732.f), g.Y(162.f) };
    g.smallR = g.S(14.f);
    g.release = { g.X(652.f), g.Y(246.f) };
    g.dry = { g.X(732.f), g.Y(246.f) };
    return g;
}

float quantize(float v, float lo, float hi, float step)
{
    v = std::clamp(v, lo, hi);
    if (step > 0.f)
        v = std::round(v / step) * step;
    return std::clamp(v, lo, hi);
}

float dragValue(float start, float dyPx, float scale, float lo, float hi, float step, bool log, bool fine)
{
    float d = dyPx / (160.f * std::max(scale, 0.6f));
    if (fine)
        d *= 0.1f;
    if (!log)
        return quantize(start + d * (hi - lo), lo, hi, step);
    const float n0 = std::log(std::max(start, lo) / lo) / std::log(hi / lo);
    const float n = std::clamp(n0 + d, 0.f, 1.f);
    return quantize(lo * std::exp(n * std::log(hi / lo)), lo, hi, step);
}

float wheelValue(float start, float delta, float lo, float hi, float step, bool log, bool fine)
{
    if (!log)
    {
        const float d = delta * (fine ? step : std::max(step, (hi - lo) * 0.04f));
        return quantize(start + d, lo, hi, step);
    }
    const float n0 = std::log(std::max(start, lo) / lo) / std::log(hi / lo);
    const float n = std::clamp(n0 + delta * (fine ? 0.01f : 0.05f), 0.f, 1.f);
    return quantize(lo * std::exp(n * std::log(hi / lo)), lo, hi, step);
}

juce::String fmtHz(float v)
{
    if (v >= 999.5f)
    {
        const float k = v / 1000.f;
        if (k >= 10.f && std::abs(k - std::round(k)) < 0.05f)
            return juce::String((int) std::round(k)) + "k";
        return juce::String(k, 1) + "k";
    }
    return juce::String((int) std::round(v)) + " Hz";
}

juce::String fmtDb(float v, bool decimal)
{
    if (v <= -95.5f)
        return "-inf";
    if (decimal)
        return std::abs(v) < 0.05f ? "0.0 dB" : juce::String(v, 1) + " dB";
    return juce::String((int) std::round(v)) + " dB";
}

juce::String fmtSt(float v, bool decimal)
{
    if (decimal)
        return std::abs(v) < 0.05f ? "0.0 st" : ((v > 0 ? "+" : "") + juce::String(v, 1) + " st");
    if (std::abs(v) < 0.5f)
        return "0 st";
    return (v > 0 ? "+" : "") + juce::String((int) std::round(v)) + " st";
}

juce::String fmtMs(float v, bool coarse)
{
    if (coarse || v >= 10.f)
        return juce::String((int) std::round(v)) + " ms";
    return juce::String(v, 1) + " ms";
}

const char* carrierName(int v)
{
    switch (v)
    {
        case 1: return "External";
        case 2: return "Modulator";
        case 3: return "Pitch Tracking";
        default: return "Noise";
    }
}

const char* waveName(int v)
{
    switch (v)
    {
        case 1: return "Pulse 50%";
        case 2: return "Pulse 25%";
        case 3: return "Pulse 12.5%";
        default: return "Saw";
    }
}

void fillRound(juce::Graphics& g, juce::Rectangle<float> r)
{
    g.fillRect(r);
}

void stroke(juce::Graphics& g, juce::Rectangle<float> r)
{
    g.drawRect(r, 1.f);
}

void drawDrop(juce::Graphics& g, juce::Rectangle<float> r, const juce::String& text, bool open, float s, const juce::Font& font)
{
    g.setColour(kPanel);
    fillRound(g, r);
    g.setColour(open ? kOrange : kTeal);
    stroke(g, r);
    g.setColour(kText);
    g.setFont(font);
    g.drawText(text, r.reduced(s * 6.f, 0.f).withTrimmedRight(s * 10.f), juce::Justification::centredLeft, false);
    juce::Path tri;
    const float cx = r.getRight() - s * 9.f;
    const float cy = r.getCentreY();
    tri.addTriangle(cx - s * 3.f, cy - s * 1.5f, cx + s * 3.f, cy - s * 1.5f, cx, cy + s * 2.2f);
    g.setColour(kDim);
    g.fillPath(tri);
}

void drawBtn(juce::Graphics& g, juce::Rectangle<float> r, const juce::String& text, bool on, const juce::Font& font)
{
    g.setColour(on ? kOrange : kPanel);
    fillRound(g, r);
    g.setColour(kLine);
    stroke(g, r);
    g.setColour(on ? kInk : kText);
    g.setFont(font);
    g.drawText(text, r, juce::Justification::centred, false);
}

void drawNum(juce::Graphics& g, juce::Rectangle<float> r, const juce::String& text, const juce::Font& font)
{
    g.setColour(kPanel);
    fillRound(g, r);
    g.setColour(kLine);
    stroke(g, r);
    g.setColour(kTeal);
    g.setFont(font);
    g.drawText(text, r, juce::Justification::centred, false);
}

void drawCaption(juce::Graphics& g, float x, float y, const juce::String& text, float fm, const juce::Font& fontMini)
{
    g.setColour(kDim);
    g.setFont(fontMini);
    g.drawText(text, juce::Rectangle<float>(x, y - fm - 2.f, 80.f, fm + 1.f), juce::Justification::bottomLeft, false);
}

void drawKnob(juce::Graphics& g, juce::Point<float> c, float r, float norm, const juce::String& cap,
              const juce::String& val, float s, float fs, float fm,
              const juce::Font& fontStd, const juce::Font& fontMini)
{
    norm = std::clamp(norm, 0.f, 1.f);
    g.setColour(kKnobC);
    g.fillEllipse(c.x - r, c.y - r, r * 2.f, r * 2.f);
    g.setColour(kLine);
    g.drawEllipse(c.x - r, c.y - r, r * 2.f, r * 2.f, 1.f);
    const float a0 = 210.f * juce::MathConstants<float>::pi / 180.f;
    const float sweep = 300.f * juce::MathConstants<float>::pi / 180.f;
    const float rad = std::max(1.f, r - std::max(1.f, s));
    juce::Path track;
    track.addCentredArc(c.x, c.y, rad, rad, 0.f, a0, a0 + sweep, true);
    g.setColour(kTrack);
    g.strokePath(track, juce::PathStrokeType(std::max(1.4f, s * 1.4f)));
    if (norm > 0.004f)
    {
        juce::Path hot;
        hot.addCentredArc(c.x, c.y, rad, rad, 0.f, a0, a0 + sweep * norm, true);
        g.setColour(kTeal);
        g.strokePath(hot, juce::PathStrokeType(std::max(1.6f, s * 1.6f)));
    }
    const float ang = a0 + sweep * norm;
    g.setColour(kTeal);
    g.drawLine(c.x + std::sin(ang) * r * 0.22f, c.y - std::cos(ang) * r * 0.22f,
               c.x + std::sin(ang) * rad, c.y - std::cos(ang) * rad,
               std::max(1.2f, s));
    g.setColour(kDim);
    g.setFont(fontMini);
    g.drawText(cap, juce::Rectangle<float>(c.x - s * 46.f, c.y - r - fm - s * 2.f, s * 92.f, fm + 1.f),
               juce::Justification::centred, false);
    g.setColour(kTeal);
    g.setFont(fontStd);
    g.drawText(val, juce::Rectangle<float>(c.x - s * 48.f, c.y + r + s, s * 96.f, fs + 2.f),
               juce::Justification::centred, false);
}

float linNorm(float v, float lo, float hi)
{
    return std::clamp((v - lo) / (hi - lo), 0.f, 1.f);
}

float logNorm(float v, float lo, float hi)
{
    v = std::clamp(v, lo, hi);
    return std::clamp(std::log(v / lo) / std::log(hi / lo), 0.f, 1.f);
}
} // namespace

AbletonVocoderEditor::AbletonVocoderEditor(AbletonVocoderProcessor& p)
    : AudioProcessorEditor(&p), proc(p)
{
    setOpaque(true);
    setWantsKeyboardFocus(true);
    setMouseClickGrabsKeyboardFocus(false);
    setResizable(true, true);
    setResizeLimits(468, 174, 2340, 870);
    if (auto* c = getConstrainer())
        c->setFixedAspectRatio(780.0 / 290.0);
    setSize(780, 290);
    for (float& db : lastDispDb)
        db = -120.f;
    startTimerHz(30);
}

AbletonVocoderEditor::~AbletonVocoderEditor() = default;

float AbletonVocoderEditor::param(const char* id) const
{
    if (auto* v = proc.apvts.getRawParameterValue(id))
        return v->load(std::memory_order_relaxed);
    return 0.f;
}

void AbletonVocoderEditor::setParam(const char* id, float value)
{
    if (auto* p = dynamic_cast<juce::RangedAudioParameter*>(proc.apvts.getParameter(id)))
        p->setValueNotifyingHost(p->convertTo0to1(value));
}

void AbletonVocoderEditor::beginParam(const char* id)
{
    if (auto* p = proc.apvts.getParameter(id))
        p->beginChangeGesture();
}

void AbletonVocoderEditor::endParam(const char* id)
{
    if (auto* p = proc.apvts.getParameter(id))
        p->endChangeGesture();
}

juce::Rectangle<float> AbletonVocoderEditor::getMenuRect() const
{
    if (menu <= 0)
        return {};
    const Geo geo = makeGeo((float) getWidth(), (float) getHeight());
    const float nf = menu == 2 ? 10.f : 4.f;
    const float ih = geo.S(18.f);
    juce::Rectangle<float> anchor = menu == 1 ? geo.carrier : menu == 2 ? geo.bands : geo.wave;
    float y = menu == 2 ? anchor.getY() - nf * ih : anchor.getBottom() + geo.S(2.f);
    float x = anchor.getX();
    float w = menu == 2 ? std::max(anchor.getWidth(), geo.S(56.f)) : anchor.getWidth();
    if (y < 2.f)
        y = 2.f;
    if (y + nf * ih > (float) getHeight() - 2.f)
        y = (float) getHeight() - 2.f - nf * ih;
    return { x, y, w, nf * ih };
}

int AbletonVocoderEditor::getMenuItemAt(juce::Point<float> p) const
{
    if (menu <= 0)
        return -1;
    const auto area = getMenuRect();
    if (!area.contains(p))
        return -1;
    const int n = menu == 2 ? 10 : 4;
    const float ih = area.getHeight() / (float) n;
    const int j = (int) std::floor((p.y - area.getY()) / std::max(ih, 1.f));
    return std::clamp(j, 0, n - 1);
}

void AbletonVocoderEditor::focusLost(FocusChangeType)
{
    if (menu != 0)
    {
        menu = 0;
        menuArmed = false;
        menuHover = -1;
        repaint();
    }
}

bool AbletonVocoderEditor::keyPressed(const juce::KeyPress& key)
{
    if (menu != 0 && key == juce::KeyPress::escapeKey)
    {
        menu = 0;
        menuArmed = false;
        menuHover = -1;
        giveAwayKeyboardFocus();
        repaint();
        return true;
    }
    return false;
}

void AbletonVocoderEditor::timerCallback()
{
    // If Wave menu is open while Carrier was automated away from Pitch Tracking, close it
    if (menu == 3 && (int) std::lround(param(voc::carrier)) != 3)
    {
        menu = 0;
        menuArmed = false;
        menuHover = -1;
        repaint();
        return;
    }

    bool dirty = false;
    const uint32_t rev = proc.stateRevision.load(std::memory_order_relaxed);
    if (rev != lastStateRev)
    {
        lastStateRev = rev;
        dirty = true;
    }

    for (int i = 0; i < 25; ++i)
    {
        const float v = param(kAllParamIds[i]);
        if (std::memcmp(&v, &lastParamSnap[i], sizeof(float)) != 0)
        {
            lastParamSnap[i] = v;
            dirty = true;
        }
    }

    bool barsDirty = false;
    const int nBands = ((int) std::lround(param(voc::bands)) + 1) * 4;
    for (int i = 0; i < nBands; ++i)
    {
        const float env = proc.dispEnv[i].load(std::memory_order_relaxed);
        const float db = env > 1.0e-12f ? std::max(-60.f, 20.f * std::log10(env)) : -60.f;
        if (std::abs(db - lastDispDb[i]) > 0.25f)
        {
            lastDispDb[i] = db;
            barsDirty = true;
        }
    }

    if (dirty)
    {
        repaint();
    }
    else if (barsDirty)
    {
        const Geo geo = makeGeo((float) getWidth(), (float) getHeight());
        repaint(geo.disp.toNearestInt().expanded(2));
    }
}

void AbletonVocoderEditor::paint(juce::Graphics& g)
{
    const Geo geo = makeGeo((float) getWidth(), (float) getHeight());
    const juce::Font fontStd(juce::FontOptions("Arial", geo.fs, juce::Font::plain));
    const juce::Font fontMini(juce::FontOptions("Arial", geo.fm, juce::Font::plain));
    const juce::Font fontBold(juce::FontOptions("Arial", geo.fb, juce::Font::bold));

    const int carrier = (int) std::lround(param(voc::carrier));
    const int bandIdx = (int) std::lround(param(voc::bands));
    const int nBands = (bandIdx + 1) * 4;
    const int channels = (int) std::lround(param(voc::channels));
    const int wave = (int) std::lround(param(voc::oscWave));
    const int fmode = (int) std::lround(param(voc::filterMode));
    const int speed = (int) std::lround(param(voc::uvSpeed));
    const bool enhance = param(voc::enhance) > 0.5f;
    const bool device = param(voc::deviceOn) > 0.5f;
    const float depth = param(voc::depth);
    const float attack = param(voc::attack);
    const float release = param(voc::release);
    const float formant = param(voc::formant);
    const float dry = param(voc::dryWet);
    const float unvoiced = param(voc::unvoiced);
    const float sens = param(voc::sens);
    const float rate = param(voc::noiseRate);
    const float crack = param(voc::noiseCrackle);

    g.fillAll(kOut);
    g.setColour(kBg);
    g.fillRect(geo.box(0, 0, 780, 290));
    g.setColour(kPanel);
    g.fillRect(geo.box(0, 0, 780, 26));
    g.setColour(kLine);
    g.drawLine(geo.X(0), geo.Y(26), geo.X(780), geo.Y(26));
    g.drawLine(geo.X(156), geo.Y(32), geo.X(156), geo.Y(276));
    g.drawLine(geo.X(606), geo.Y(32), geo.X(606), geo.Y(276));

    g.setColour(device ? kOrange : kKnobC);
    g.fillEllipse(geo.power.x - geo.powerR, geo.power.y - geo.powerR, geo.powerR * 2.f, geo.powerR * 2.f);
    if (!device)
    {
        g.setColour(kDim);
        g.drawEllipse(geo.power.x - geo.powerR, geo.power.y - geo.powerR, geo.powerR * 2.f, geo.powerR * 2.f, 1.f);
    }
    g.setColour(kText);
    g.setFont(fontBold);
    g.drawText("Vocoder", juce::Rectangle<float>(geo.X(30), geo.Y(4), geo.S(160), geo.S(20)),
               juce::Justification::centredLeft, false);

    drawCaption(g, geo.carrier.getX(), geo.carrier.getY(), "Carrier", geo.fm, fontMini);
    drawDrop(g, geo.carrier, carrierName(carrier), menu == 1, geo.s, fontStd);
    g.setColour(kPanel);
    g.fillRect(geo.ctx);
    g.setColour(kLine);
    stroke(g, geo.ctx);

    if (carrier == 0)
    {
        g.setColour(kScreen);
        g.fillRect(geo.pad);
        g.setColour(kLine);
        stroke(g, geo.pad);
        g.setColour(kDim.withAlpha(0.28f));
        g.drawLine(geo.pad.getCentreX(), geo.pad.getY() + geo.S(3), geo.pad.getCentreX(), geo.pad.getBottom() - geo.S(3));
        g.drawLine(geo.pad.getX() + geo.S(3), geo.pad.getCentreY(), geo.pad.getRight() - geo.S(3), geo.pad.getCentreY());
        g.setColour(kDim);
        g.setFont(fontMini);
        g.drawText("Crackle " + juce::String(std::clamp(crack, 0.f, 1.f), 2),
                   geo.pad.reduced(geo.S(4), geo.S(3)).removeFromTop(geo.fm + 2.f), juce::Justification::topLeft, false);
        g.drawText("Rate " + juce::String(std::clamp(rate, 0.f, 1.f), 2),
                   geo.pad.reduced(geo.S(4), geo.S(3)).removeFromBottom(geo.fm + 2.f), juce::Justification::bottomLeft, false);
        const float rad = std::max(3.f, geo.S(4.5f));
        const float span = rad + geo.S(2.f);
        const float dx = geo.pad.getX() + span + std::clamp(rate, 0.f, 1.f) * std::max(1.f, geo.pad.getWidth() - span * 2.f);
        const float dy = geo.pad.getY() + span + (1.f - std::clamp(crack, 0.f, 1.f)) * std::max(1.f, geo.pad.getHeight() - span * 2.f);
        g.setColour(kKnobC);
        g.fillEllipse(dx - rad, dy - rad, rad * 2.f, rad * 2.f);
        g.setColour(kTeal);
        g.fillEllipse(dx - rad * 0.62f, dy - rad * 0.62f, rad * 1.24f, rad * 1.24f);
    }
    else if (carrier == 3)
    {
        drawDrop(g, geo.wave, waveName(wave), menu == 3, geo.s, fontStd);
        auto row = [&](juce::Rectangle<float> r, const juce::String& name, const juce::String& value)
        {
            g.setColour(kDim);
            g.setFont(fontMini);
            g.drawText(name, r.reduced(geo.S(4), 0).removeFromLeft(geo.S(42)), juce::Justification::centredLeft, false);
            g.setColour(kTeal);
            g.setFont(fontStd);
            g.drawText(value, r.reduced(geo.S(4), 0), juce::Justification::centredRight, false);
        };
        row(geo.pitch, "Pitch", fmtSt(param(voc::oscPitch), false));
        row(geo.low, "Low", fmtHz(param(voc::trackLow)));
        row(geo.high, "High", fmtHz(param(voc::trackHigh)));
    }
    else
    {
        g.setColour(kText);
        g.setFont(fontStd);
        g.drawFittedText(carrier == 1 ? "Sidechain: ch 3/4" : "Carrier = modulator",
                         geo.ctx.reduced(geo.S(6)).toNearestInt(), juce::Justification::centred, 2);
    }

    drawBtn(g, geo.enhance, "Enhance", enhance, fontStd);
    drawKnob(g, geo.uv, geo.uvR, linNorm(unvoiced, -96.f, 0.f), "Unvoiced", fmtDb(unvoiced, false), geo.s, geo.fs, geo.fm, fontStd, fontMini);
    drawCaption(g, geo.sens.getX(), geo.sens.getY(), "Sens.", geo.fm, fontMini);
    drawNum(g, geo.sens, juce::String((int) std::round(sens)) + "%", fontStd);
    drawBtn(g, geo.speed, speed > 0 ? "Fast" : "Slow", speed > 0, fontStd);

    g.setColour(kScreen);
    g.fillRect(geo.disp);
    g.setColour(kLine);
    stroke(g, geo.disp);
    const float gap = nBands >= 24 ? 1.f : 2.f;
    for (int i = 0; i < nBands; ++i)
    {
        const float x1 = std::floor(geo.bars.getX() + (float) i * geo.bars.getWidth() / (float) nBands);
        const float x2 = std::floor(geo.bars.getX() + (float) (i + 1) * geo.bars.getWidth() / (float) nBands);
        const float bw = std::max(1.f, x2 - x1 - gap);
        const float env = proc.dispEnv[i].load(std::memory_order_relaxed);
        const float db = env > 1.0e-12f ? 20.f * std::log10(env) : -120.f;
        const float t = std::clamp((db + 60.f) / 60.f, 0.f, 1.f);
        const float bh = t * geo.bars.getHeight();
        if (bh > 0.5f)
        {
            g.setColour(kTeal);
            g.fillRect(x1, geo.bars.getBottom() - bh, bw, bh);
        }
        const float att = std::clamp(proc.getBandAttDb(i), -60.f, 0.f);
        if (att < -0.05f)
        {
            const float my = geo.bars.getY() + geo.bars.getHeight() * (-att / 60.f);
            g.setColour(juce::Colour(255, 140, 24).withAlpha(0.32f));
            if (my - geo.bars.getY() > 0.5f)
                g.fillRect(x1, geo.bars.getY(), bw, my - geo.bars.getY());
            g.setColour(kOrange);
            g.drawLine(x1, my, x1 + bw, my, 1.f);
        }
    }

    drawCaption(g, geo.bands.getX(), geo.bands.getY(), "Bands", geo.fm, fontMini);
    drawDrop(g, geo.bands, juce::String(nBands), menu == 2, geo.s, fontStd);
    drawCaption(g, geo.rLow.getX(), geo.rLow.getY(), "Range", geo.fm, fontMini);
    drawNum(g, geo.rLow, fmtHz(param(voc::rangeLow)), fontStd);
    drawNum(g, geo.rHigh, fmtHz(param(voc::rangeHigh)), fontStd);
    drawCaption(g, geo.bw.getX(), geo.bw.getY(), "BW", geo.fm, fontMini);
    drawNum(g, geo.bw, juce::String((int) std::round(param(voc::bw))) + "%", fontStd);
    drawCaption(g, geo.mode.getX(), geo.mode.getY(), fmode ? "Retro" : "Precise", geo.fm, fontMini);
    drawBtn(g, geo.mode, fmode ? "Retro" : "Precise", fmode != 0, fontStd);
    drawCaption(g, geo.gate.getX(), geo.gate.getY(), "Gate", geo.fm, fontMini);
    drawNum(g, geo.gate, fmtDb(param(voc::gate), false), fontStd);
    drawCaption(g, geo.level.getX(), geo.level.getY(), "Level", geo.fm, fontMini);
    drawNum(g, geo.level, fmtDb(param(voc::level), true), fontStd);

    drawKnob(g, geo.depth, geo.depthR, linNorm(depth, 0.f, 200.f), "Depth",
             juce::String((int) std::round(depth)) + "%", geo.s, geo.fs, geo.fm, fontStd, fontMini);
    drawBtn(g, geo.chMono, "Mono", channels == 0, fontStd);
    drawBtn(g, geo.chSt, "Stereo", channels == 1, fontStd);
    drawBtn(g, geo.chLR, "L/R", channels == 2, fontStd);
    drawKnob(g, geo.attack, geo.smallR, logNorm(attack, 0.1f, 100.f), "Attack", fmtMs(attack, false), geo.s, geo.fs, geo.fm, fontStd, fontMini);
    drawKnob(g, geo.formant, geo.smallR, linNorm(formant, -12.f, 12.f), "Formant", fmtSt(formant, true), geo.s, geo.fs, geo.fm, fontStd, fontMini);
    drawKnob(g, geo.release, geo.smallR, logNorm(release, 1.f, 1000.f), "Release", fmtMs(release, true), geo.s, geo.fs, geo.fm, fontStd, fontMini);
    drawKnob(g, geo.dry, geo.smallR, linNorm(dry, 0.f, 100.f), "Dry/Wet",
             juce::String((int) std::round(dry)) + "%", geo.s, geo.fs, geo.fm, fontStd, fontMini);

    if (menu > 0)
    {
        const auto area = getMenuRect();
        const int n = menu == 2 ? 10 : 4;
        const float ih = area.getHeight() / (float) n;
        const float x = area.getX();
        const float y = area.getY();
        const float w = area.getWidth();
        g.setColour(juce::Colours::black.withAlpha(0.4f));
        g.fillRect(x + geo.S(2), y + geo.S(3), w, area.getHeight());
        g.setColour(kPanel);
        g.fillRect(area);
        g.setColour(kTeal);
        g.drawRect(area, 1.f);
        for (int j = 0; j < n; ++j)
        {
            const auto row = juce::Rectangle<float>(x, y + (float) j * ih, w, ih);
            const bool hov = (j == menuHover);
            const bool cur = (menu == 1 && carrier == j) || (menu == 2 && bandIdx == j) || (menu == 3 && wave == j);
            if (hov)
            {
                g.setColour(kTeal.withAlpha(0.25f));
                g.fillRect(row.reduced(1.f, 0.f));
            }
            if (cur)
            {
                g.setColour(kOrange);
                g.fillRect(row.getX() + geo.S(2), row.getY() + row.getHeight() * 0.22f,
                           std::max(2.f, geo.S(3.f)), row.getHeight() * 0.56f);
            }
            juce::String label = menu == 1 ? carrierName(j) : menu == 2 ? juce::String((j + 1) * 4) : waveName(j);
            g.setColour(hov ? kText : cur ? kTeal : kDim);
            g.setFont(fontStd);
            g.drawText(label, row.reduced(geo.S(8), 0), juce::Justification::centredLeft, false);
        }
    }
}

void AbletonVocoderEditor::mouseMove(const juce::MouseEvent& e)
{
    if (menu > 0)
    {
        const int hov = getMenuItemAt(e.position);
        if (hov != menuHover)
        {
            menuHover = hov;
            repaint();
        }
    }
}

void AbletonVocoderEditor::mouseExit(const juce::MouseEvent&)
{
    if (menu > 0 && menuHover != -1)
    {
        menuHover = -1;
        repaint();
    }
}

void AbletonVocoderEditor::mouseDown(const juce::MouseEvent& e)
{
    const Geo geo = makeGeo((float) getWidth(), (float) getHeight());
    const auto p = e.position;
    const bool dbl = e.getNumberOfClicks() >= 2;

    if (menu > 0 && menuArmed)
    {
        const int item = getMenuItemAt(p);
        if (item >= 0)
        {
            const char* id = menu == 1 ? voc::carrier : menu == 2 ? voc::bands : voc::oscWave;
            beginParam(id);
            setParam(id, (float) item);
            endParam(id);
        }
        menu = 0;
        menuArmed = false;
        menuHover = -1;
        giveAwayKeyboardFocus();
        repaint();
        return;
    }

    auto toggle = [&](const char* id, float value)
    {
        beginParam(id);
        setParam(id, value);
        endParam(id);
        repaint();
    };
    auto start = [&](const char* id, float lo, float hi, float step, bool log)
    {
        if (dbl)
        {
            beginParam(id);
            setParam(id, id == voc::attack ? 5.f
                          : id == voc::release ? 80.f
                          : id == voc::formant ? 0.f
                          : id == voc::depth ? 100.f
                          : id == voc::dryWet ? 100.f
                          : id == voc::unvoiced ? -96.f
                          : id == voc::sens ? 50.f
                          : id == voc::bw ? 100.f
                          : id == voc::gate ? -96.f
                          : id == voc::level ? 0.f
                          : id == voc::rangeLow ? 80.f
                          : id == voc::rangeHigh ? 18000.f
                          : id == voc::oscPitch ? 0.f
                          : id == voc::trackLow ? 60.f
                          : id == voc::trackHigh ? 1000.f
                          : param(id));
            endParam(id);
            repaint();
            return;
        }
        drag = Drag::value;
        dragId = id;
        dragLo = lo;
        dragHi = hi;
        dragStep = step;
        dragLog = log;
        dragStart = param(id);
        dragY = p.y;
        dragMoved = false;
        beginParam(id);
    };

    if (p.getDistanceFrom(geo.power) <= geo.powerR + geo.S(5.f))
    {
        toggle(voc::deviceOn, param(voc::deviceOn) > 0.5f ? 0.f : 1.f);
        return;
    }
    auto openMenu = [&](int mId)
    {
        menu = mId;
        menuArmed = false;
        menuDownPos = p;
        menuDownMs = juce::Time::getMillisecondCounter();
        menuHover = getMenuItemAt(p);
        grabKeyboardFocus();
        repaint();
    };
    if (geo.carrier.contains(p)) { openMenu(1); return; }
    if (geo.bands.contains(p)) { openMenu(2); return; }
    if ((int) std::lround(param(voc::carrier)) == 3 && geo.wave.contains(p))
    {
        openMenu(3);
        return;
    }
    if (geo.enhance.contains(p)) { toggle(voc::enhance, param(voc::enhance) > 0.5f ? 0.f : 1.f); return; }
    if (geo.speed.contains(p)) { toggle(voc::uvSpeed, param(voc::uvSpeed) > 0.5f ? 0.f : 1.f); return; }
    if (geo.mode.contains(p)) { toggle(voc::filterMode, param(voc::filterMode) > 0.5f ? 0.f : 1.f); return; }
    if (geo.chMono.contains(p)) { toggle(voc::channels, 0.f); return; }
    if (geo.chSt.contains(p)) { toggle(voc::channels, 1.f); return; }
    if (geo.chLR.contains(p)) { toggle(voc::channels, 2.f); return; }

    if (geo.around(geo.depth, geo.depthR).contains(p)) { start(voc::depth, 0, 200, 1, false); return; }
    if (geo.around(geo.attack, geo.smallR).contains(p)) { start(voc::attack, 0.1f, 100, 0.1f, true); return; }
    if (geo.around(geo.release, geo.smallR).contains(p)) { start(voc::release, 1, 1000, 1, true); return; }
    if (geo.around(geo.formant, geo.smallR).contains(p)) { start(voc::formant, -12, 12, 0.1f, false); return; }
    if (geo.around(geo.dry, geo.smallR).contains(p)) { start(voc::dryWet, 0, 100, 1, false); return; }
    if (geo.around(geo.uv, geo.uvR).contains(p)) { start(voc::unvoiced, -96, 0, 1, false); return; }
    if (geo.sens.contains(p)) { start(voc::sens, 0, 100, 1, false); return; }
    if (geo.bw.contains(p)) { start(voc::bw, 10, 200, 1, false); return; }
    if (geo.gate.contains(p)) { start(voc::gate, -96, 0, 1, false); return; }
    if (geo.level.contains(p)) { start(voc::level, -24, 24, 0.1f, false); return; }
    if (geo.rLow.contains(p)) { start(voc::rangeLow, 20, 2000, 1, true); return; }
    if (geo.rHigh.contains(p)) { start(voc::rangeHigh, 1000, 20000, 1, true); return; }

    if ((int) std::lround(param(voc::carrier)) == 0 && geo.pad.contains(p))
    {
        if (dbl)
        {
            beginParam(voc::noiseRate); setParam(voc::noiseRate, 1.f); endParam(voc::noiseRate);
            beginParam(voc::noiseCrackle); setParam(voc::noiseCrackle, 1.f); endParam(voc::noiseCrackle);
            repaint();
            return;
        }
        drag = Drag::xy;
        xyRate = xyCrack = false;
        beginParam(voc::noiseRate);
        beginParam(voc::noiseCrackle);
        mouseDrag(e);
        return;
    }
    if ((int) std::lround(param(voc::carrier)) == 3)
    {
        if (geo.pitch.contains(p)) { start(voc::oscPitch, -24, 24, 1, false); return; }
        if (geo.low.contains(p)) { start(voc::trackLow, 20, 1000, 1, true); return; }
        if (geo.high.contains(p)) { start(voc::trackHigh, 100, 2000, 1, true); return; }
    }

    if (geo.disp.contains(p))
    {
        const int nBands = ((int) std::lround(param(voc::bands)) + 1) * 4;
        const float t = std::clamp((p.x - geo.bars.getX()) / std::max(geo.bars.getWidth(), 1.f), 0.f, 0.999999f);
        const int band = std::clamp((int) std::floor(t * (float) nBands), 0, nBands - 1);
        if (e.mods.isAltDown())
        {
            proc.resetAllBands();
            proc.updateHostDisplay(juce::AudioProcessorListener::ChangeDetails().withNonParameterStateChanged(true));
            repaint();
            return;
        }
        if (dbl)
        {
            proc.setBandAttDb(band, 0.f);
            proc.updateHostDisplay(juce::AudioProcessorListener::ChangeDetails().withNonParameterStateChanged(true));
            repaint();
            return;
        }
        const float att = quantize(-60.f * std::clamp((p.y - geo.bars.getY()) / std::max(geo.bars.getHeight(), 1.f), 0.f, 1.f), -60.f, 0.f, 0.1f);
        drag = Drag::band;
        bandLast = band;
        bandLastAtt = att;
        bandDirty = false;
        if (std::abs(proc.getBandAttDb(band) - att) > 0.05f)
        {
            proc.setBandAttDb(band, att);
            bandDirty = true;
            repaint();
        }
    }
}

void AbletonVocoderEditor::mouseDrag(const juce::MouseEvent& e)
{
    if (menu > 0)
    {
        const int hov = getMenuItemAt(e.position);
        if (hov != menuHover)
        {
            menuHover = hov;
            repaint();
        }
        return;
    }

    const Geo geo = makeGeo((float) getWidth(), (float) getHeight());
    if (drag == Drag::value && dragId != nullptr)
    {
        const float v = dragValue(dragStart, dragY - e.position.y, geo.s, dragLo, dragHi, dragStep, dragLog, e.mods.isShiftDown());
        if (std::abs(v - param(dragId)) > dragStep * 0.25f)
        {
            setParam(dragId, v);
            dragMoved = true;
            repaint();
        }
        return;
    }
    if (drag == Drag::xy)
    {
        float nx = std::clamp((e.position.x - geo.pad.getX()) / std::max(geo.pad.getWidth(), 1.f), 0.f, 1.f);
        float ny = std::clamp(1.f - (e.position.y - geo.pad.getY()) / std::max(geo.pad.getHeight(), 1.f), 0.f, 1.f);
        nx = quantize(nx, 0.f, 1.f, 0.01f);
        ny = quantize(ny, 0.f, 1.f, 0.01f);
        if (std::abs(nx - param(voc::noiseRate)) >= 0.005f)
        {
            setParam(voc::noiseRate, nx);
            xyRate = true;
            repaint();
        }
        if (std::abs(ny - param(voc::noiseCrackle)) >= 0.005f)
        {
            setParam(voc::noiseCrackle, ny);
            xyCrack = true;
            repaint();
        }
        return;
    }
    if (drag == Drag::band)
    {
        const int nBands = ((int) std::lround(param(voc::bands)) + 1) * 4;
        const float tx = std::clamp((e.position.x - geo.bars.getX()) / std::max(geo.bars.getWidth(), 1.f), 0.f, 0.999999f);
        const int band = std::clamp((int) std::floor(tx * (float) nBands), 0, nBands - 1);
        const float att = quantize(-60.f * std::clamp((e.position.y - geo.bars.getY()) / std::max(geo.bars.getHeight(), 1.f), 0.f, 1.f), -60.f, 0.f, 0.1f);
        const int i0 = bandLast < 0 ? band : bandLast;
        const float a0 = bandLast < 0 ? att : bandLastAtt;
        const int dir = band >= i0 ? 1 : -1;
        const int span = std::abs(band - i0);
        for (int k = 0; k <= span; ++k)
        {
            const int i = i0 + dir * k;
            const float a = span == 0 ? att : a0 + (att - a0) * (float) (i - i0) / (float) (band - i0);
            const float q = quantize(a, -60.f, 0.f, 0.1f);
            if (std::abs(proc.getBandAttDb(i) - q) > 0.05f)
            {
                proc.setBandAttDb(i, q);
                bandDirty = true;
            }
        }
        bandLast = band;
        bandLastAtt = att;
        if (bandDirty)
            repaint();
    }
}

void AbletonVocoderEditor::mouseUp(const juce::MouseEvent& e)
{
    if (drag == Drag::value && dragId != nullptr)
        endParam(dragId);
    if (drag == Drag::xy)
    {
        endParam(voc::noiseRate);
        endParam(voc::noiseCrackle);
    }
    if (drag == Drag::band && bandDirty)
        proc.updateHostDisplay(juce::AudioProcessorListener::ChangeDetails().withNonParameterStateChanged(true));
    if (menu > 0 && !menuArmed)
    {
        const int item = getMenuItemAt(e.position);
        const float dist = e.position.getDistanceFrom(menuDownPos);
        const uint32_t heldMs = juce::Time::getMillisecondCounter() - menuDownMs;
        if (item >= 0 && (dist > 3.f || heldMs > 220u))
        {
            // Support click-hold-drag-release selection on dropdown menus
            const char* id = menu == 1 ? voc::carrier : menu == 2 ? voc::bands : voc::oscWave;
            beginParam(id);
            setParam(id, (float) item);
            endParam(id);
            menu = 0;
            menuArmed = false;
            menuHover = -1;
            giveAwayKeyboardFocus();
            repaint();
        }
        else
        {
            menuArmed = true;
        }
    }
    drag = Drag::none;
    dragId = nullptr;
    bandLast = -1;
    bandDirty = false;
    xyRate = xyCrack = false;
}

void AbletonVocoderEditor::mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel)
{
    const float delta = std::abs(wheel.deltaY) > std::abs(wheel.deltaX) ? wheel.deltaY : wheel.deltaX;
    if (menu > 0 || delta == 0.f)
        return;
    const Geo geo = makeGeo((float) getWidth(), (float) getHeight());
    const auto p = e.position;
    const bool fine = e.mods.isShiftDown();
    const char* id = nullptr;
    float lo = 0, hi = 1, step = 0.01f;
    bool log = false;
    auto hit = [&](bool cond, const char* pid, float a, float b, float st, bool lg)
    {
        if (id == nullptr && cond)
        {
            id = pid; lo = a; hi = b; step = st; log = lg;
        }
    };
    hit(geo.around(geo.depth, geo.depthR).contains(p), voc::depth, 0, 200, 1, false);
    hit(geo.around(geo.attack, geo.smallR).contains(p), voc::attack, 0.1f, 100, 0.1f, true);
    hit(geo.around(geo.release, geo.smallR).contains(p), voc::release, 1, 1000, 1, true);
    hit(geo.around(geo.formant, geo.smallR).contains(p), voc::formant, -12, 12, 0.1f, false);
    hit(geo.around(geo.dry, geo.smallR).contains(p), voc::dryWet, 0, 100, 1, false);
    hit(geo.around(geo.uv, geo.uvR).contains(p), voc::unvoiced, -96, 0, 1, false);
    hit(geo.sens.contains(p), voc::sens, 0, 100, 1, false);
    hit(geo.bw.contains(p), voc::bw, 10, 200, 1, false);
    hit(geo.gate.contains(p), voc::gate, -96, 0, 1, false);
    hit(geo.level.contains(p), voc::level, -24, 24, 0.1f, false);
    hit(geo.rLow.contains(p), voc::rangeLow, 20, 2000, 1, true);
    hit(geo.rHigh.contains(p), voc::rangeHigh, 1000, 20000, 1, true);
    if ((int) std::lround(param(voc::carrier)) == 3)
    {
        hit(geo.pitch.contains(p), voc::oscPitch, -24, 24, 1, false);
        hit(geo.low.contains(p), voc::trackLow, 20, 1000, 1, true);
        hit(geo.high.contains(p), voc::trackHigh, 100, 2000, 1, true);
    }
    if (id == nullptr && (int) std::lround(param(voc::carrier)) == 0 && geo.pad.contains(p))
    {
        id = fine ? voc::noiseRate : voc::noiseCrackle;
        lo = 0; hi = 1; step = 0.01f; log = false;
    }
    if (id == nullptr)
        return;
    const float cur = param(id);
    float next = wheelValue(cur, delta, lo, hi, step, log, fine);
    if (std::memcmp(&next, &cur, sizeof(float)) == 0 && delta != 0.f && step > 0.f)
        next = quantize(cur + (delta > 0.f ? step : -step), lo, hi, step);
    beginParam(id);
    setParam(id, next);
    endParam(id);
    repaint();
}
