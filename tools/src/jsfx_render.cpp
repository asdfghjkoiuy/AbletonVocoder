// Headless JSFX renderer.
//
//   jsfx_render <plugin.jsfx> <in.wav> <out.wav>
//       [--sr N] [--block N] [--set IDX=VAL ...]
//       [--automate file.csv] [--serialize-dump]
//
// in.wav is 2 or 4 channels (mod L/R, optional carrier L/R). The plugin always
// receives 4 inputs; missing carrier channels are zero. out.wav is 2-channel
// 32-bit float. Slider IDX is 1-based. Automation CSV rows are
// sample_offset,slider_idx,value and split the processed block so @slider runs
// before @block/@sample at that sample, the same way REAPER does.
// @gfx is not compiled and not run.
//
// stderr gets compile errors, wall time, realtime factor, and NaN/Inf counts.
// A compile or load failure exits non-zero. A finished render exits 0 even if
// the output contains NaN (the counts are printed).

#include "ysfx.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

void enable_ftz() {
#if defined(__aarch64__)
    // Match a DAW audio thread: flush denormal results so the CPU figure is
    // about the algorithm, not about denormal stalls.
    uint64_t fpcr = 0;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ull << 24);
    __asm__ __volatile__("msr fpcr, %0" ::"r"(fpcr));
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int csr = 0;
    __asm__ __volatile__("stmxcsr %0" : "=m"(csr));
    csr |= 0x8040u;  // FTZ | DAZ
    __asm__ __volatile__("ldmxcsr %0" ::"m"(csr));
#else
    (void)0;
#endif
}

struct Args {
    std::string plugin;
    std::string input;
    std::string output;
    int sr = 0;
    int block = 512;
    std::vector<std::pair<int, double>> sets;
    std::string automate;
    bool serialize_dump = false;
};

void usage() {
    std::fprintf(stderr,
                 "Usage: jsfx_render <plugin.jsfx> <in.wav> <out.wav> "
                 "[--sr N] [--block N] [--set IDX=VAL ...] "
                 "[--automate file.csv] [--serialize-dump]\n");
}

bool parse_set(const std::string &text, int &idx, double &value) {
    const auto eq = text.find('=');
    if (eq == std::string::npos || eq == 0) return false;
    try {
        idx = std::stoi(text.substr(0, eq));
        value = std::stod(text.substr(eq + 1));
    } catch (...) {
        return false;
    }
    return idx >= 1 && idx <= ysfx_max_sliders && std::isfinite(value);
}

bool parse_args(int argc, char **argv, Args &args) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--sr") {
            const char *v = need("--sr");
            if (!v) return false;
            args.sr = std::atoi(v);
            if (args.sr < 1000) {
                std::fprintf(stderr, "invalid --sr\n");
                return false;
            }
        } else if (a == "--block") {
            const char *v = need("--block");
            if (!v) return false;
            args.block = std::atoi(v);
            if (args.block < 1 || args.block > 1 << 20) {
                std::fprintf(stderr, "invalid --block\n");
                return false;
            }
        } else if (a == "--set") {
            const char *v = need("--set");
            if (!v) return false;
            int idx = 0;
            double value = 0;
            if (!parse_set(v, idx, value)) {
                std::fprintf(stderr, "bad --set '%s' (want IDX=VAL, IDX >= 1)\n", v);
                return false;
            }
            args.sets.emplace_back(idx, value);
        } else if (a == "--automate") {
            const char *v = need("--automate");
            if (!v) return false;
            args.automate = v;
        } else if (a == "--serialize-dump") {
            args.serialize_dump = true;
        } else if (a == "--help" || a == "-h") {
            usage();
            std::exit(0);
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return false;
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 3) {
        usage();
        return false;
    }
    args.plugin = pos[0];
    args.input = pos[1];
    args.output = pos[2];
    return true;
}

uint16_t rd_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
void wr_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
}
void wr_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

struct Wav {
    int sr = 0;
    int channels = 0;
    std::vector<float> interleaved;
};

bool read_wav(const std::string &path, Wav &wav) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open input %s\n", path.c_str());
        return false;
    }
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (file.size() < 12 || std::memcmp(file.data(), "RIFF", 4) != 0 || std::memcmp(file.data() + 8, "WAVE", 4) != 0) {
        std::fprintf(stderr, "not a RIFF/WAVE file: %s\n", path.c_str());
        return false;
    }
    int format = 0;
    int channels = 0;
    int sr = 0;
    int bits = 0;
    int valid_bits = 0;
    const uint8_t *data = nullptr;
    size_t data_bytes = 0;
    size_t pos = 12;
    while (pos + 8 <= file.size()) {
        const uint8_t *chunk = file.data() + pos;
        const uint32_t sz = rd_u32(chunk + 4);
        const size_t body = pos + 8;
        if (body + sz > file.size()) {
            std::fprintf(stderr, "truncated wav chunk\n");
            return false;
        }
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (sz < 16) {
                std::fprintf(stderr, "short fmt chunk\n");
                return false;
            }
            format = rd_u16(file.data() + body);
            channels = rd_u16(file.data() + body + 2);
            sr = (int)rd_u32(file.data() + body + 4);
            bits = rd_u16(file.data() + body + 14);
            valid_bits = bits;
            if (format == 0xFFFE) {
                if (sz < 40) {
                    std::fprintf(stderr, "short extensible fmt\n");
                    return false;
                }
                valid_bits = rd_u16(file.data() + body + 18);
                format = rd_u16(file.data() + body + 24);
            }
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data = file.data() + body;
            data_bytes = sz;
        }
        pos = body + sz + (sz & 1u);
    }
    if (!data || channels < 1 || sr < 1000 || bits < 8) {
        std::fprintf(stderr, "wav is missing fmt/data or has a bad rate\n");
        return false;
    }
    if (channels != 2 && channels != 4) {
        std::fprintf(stderr, "input must have 2 or 4 channels (got %d)\n", channels);
        return false;
    }
    const int bytes_per = bits / 8;
    if (bytes_per < 1 || data_bytes / (size_t)(bytes_per * channels) == 0) {
        std::fprintf(stderr, "wav has no frames\n");
        return false;
    }
    const size_t frames = data_bytes / (size_t)(bytes_per * channels);
    wav.sr = sr;
    wav.channels = channels;
    wav.interleaved.assign(frames * (size_t)channels, 0.f);
    const uint8_t *p = data;
    for (size_t i = 0; i < frames * (size_t)channels; ++i) {
        float s = 0.f;
        if (format == 1 && bits == 16) {
            int16_t v = (int16_t)rd_u16(p);
            s = (float)v / 32768.f;
        } else if (format == 1 && bits == 24) {
            int32_t v = (int32_t)(p[0] | (p[1] << 8) | (p[2] << 16));
            if (v & 0x800000) v |= ~0xFFFFFF;
            s = (float)v / 8388608.f;
        } else if (format == 1 && bits == 32) {
            int32_t v = (int32_t)rd_u32(p);
            s = (float)((double)v / 2147483648.0);
        } else if (format == 3 && bits == 32) {
            uint32_t u = rd_u32(p);
            std::memcpy(&s, &u, 4);
        } else if (format == 3 && bits == 64) {
            uint64_t u = 0;
            for (int b = 0; b < 8; ++b) u |= (uint64_t)p[b] << (8 * b);
            double d = 0;
            std::memcpy(&d, &u, 8);
            s = (float)d;
        } else {
            std::fprintf(stderr, "unsupported wav format tag %d bits %d\n", format, bits);
            return false;
        }
        (void)valid_bits;
        wav.interleaved[i] = s;
        p += bytes_per;
    }
    return true;
}

bool write_wav_f32(const std::string &path, int sr, const std::vector<float> &interleaved, int channels) {
    if (channels != 2 || sr < 1000) return false;
    const uint32_t frames = (uint32_t)(interleaved.size() / (size_t)channels);
    const uint32_t data_bytes = frames * (uint32_t)channels * 4u;
    const uint32_t riff_size = 36u + data_bytes;
    uint8_t hdr[44];
    std::memset(hdr, 0, sizeof(hdr));
    std::memcpy(hdr + 0, "RIFF", 4);
    wr_u32(hdr + 4, riff_size);
    std::memcpy(hdr + 8, "WAVE", 4);
    std::memcpy(hdr + 12, "fmt ", 4);
    wr_u32(hdr + 16, 16);
    wr_u16(hdr + 20, 3);  // IEEE float
    wr_u16(hdr + 22, (uint16_t)channels);
    wr_u32(hdr + 24, (uint32_t)sr);
    wr_u32(hdr + 28, (uint32_t)sr * (uint32_t)channels * 4u);
    wr_u16(hdr + 32, (uint16_t)(channels * 4));
    wr_u16(hdr + 34, 32);
    std::memcpy(hdr + 36, "data", 4);
    wr_u32(hdr + 40, data_bytes);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return false;
    }
    out.write(reinterpret_cast<const char *>(hdr), 44);
    out.write(reinterpret_cast<const char *>(interleaved.data()), (std::streamsize)data_bytes);
    return (bool)out;
}

void resample_linear(const Wav &in, int out_sr, Wav &out) {
    if (in.sr == out_sr) {
        out = in;
        return;
    }
    const double ratio = (double)in.sr / (double)out_sr;
    const size_t in_frames = in.interleaved.size() / (size_t)in.channels;
    const size_t out_frames = std::max<size_t>(1, (size_t)std::llround((double)in_frames * (double)out_sr / (double)in.sr));
    out.sr = out_sr;
    out.channels = in.channels;
    out.interleaved.assign(out_frames * (size_t)in.channels, 0.f);
    for (size_t i = 0; i < out_frames; ++i) {
        const double src = (double)i * ratio;
        size_t i0 = (size_t)src;
        if (i0 >= in_frames) i0 = in_frames - 1;
        const size_t i1 = std::min(i0 + 1, in_frames - 1);
        const float f = (float)(src - (double)i0);
        for (int c = 0; c < in.channels; ++c) {
            const float a = in.interleaved[i0 * (size_t)in.channels + (size_t)c];
            const float b = in.interleaved[i1 * (size_t)in.channels + (size_t)c];
            out.interleaved[i * (size_t)in.channels + (size_t)c] = a + (b - a) * f;
        }
    }
    std::fprintf(stderr, "resample: %d -> %d\n", in.sr, out_sr);
}

struct Event {
    int64_t sample = 0;
    int slider = 1;  // 1-based
    double value = 0;
};

bool load_automation(const std::string &path, std::vector<Event> &events) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open automation %s\n", path.c_str());
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t b = 0;
        while (b < line.size() && std::isspace((unsigned char)line[b])) ++b;
        if (b >= line.size() || line[b] == '#') continue;
        std::replace(line.begin(), line.end(), ';', ',');
        std::stringstream ss(line.substr(b));
        std::string a, c, d;
        if (!std::getline(ss, a, ',') || !std::getline(ss, c, ',') || !std::getline(ss, d, ',')) {
            if (lineno == 1) continue;
            std::fprintf(stderr, "bad automation line %d\n", lineno);
            return false;
        }
        try {
            Event e;
            e.sample = std::stoll(a);
            e.slider = std::stoi(c);
            e.value = std::stod(d);
            if (e.slider < 1 || e.slider > ysfx_max_sliders || !std::isfinite(e.value)) {
                std::fprintf(stderr, "bad automation line %d\n", lineno);
                return false;
            }
            events.push_back(e);
        } catch (...) {
            if (lineno == 1) continue;
            std::fprintf(stderr, "bad automation line %d\n", lineno);
            return false;
        }
    }
    std::stable_sort(events.begin(), events.end(), [](const Event &x, const Event &y) {
        if (x.sample != y.sample) return x.sample < y.sample;
        return x.slider < y.slider;
    });
    return true;
}

void log_reporter(intptr_t, ysfx_log_level level, const char *message) {
    std::fprintf(stderr, "ysfx %s: %s\n", ysfx_log_level_string(level), message);
}

void apply_slider(ysfx_t *fx, int index_1based, double value) {
    ysfx_slider_set_value(fx, (uint32_t)(index_1based - 1), (ysfx_real)value, true);
}

void hex_dump(const uint8_t *data, size_t n) {
    static const char *hexd = "0123456789abcdef";
    std::string line;
    line.reserve(n * 2 + 1);
    for (size_t i = 0; i < n; ++i) {
        line.push_back(hexd[data[i] >> 4]);
        line.push_back(hexd[data[i] & 0xf]);
    }
    std::fputs(line.c_str(), stdout);
    std::fputc('\n', stdout);
}

}  // namespace

int main(int argc, char **argv) {
    enable_ftz();
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    Wav input;
    if (!read_wav(args.input, input)) return 2;
    const int sr = args.sr > 0 ? args.sr : input.sr;
    if (sr != input.sr) {
        Wav resampled;
        resample_linear(input, sr, resampled);
        input = std::move(resampled);
    }

    std::vector<Event> events;
    if (!args.automate.empty() && !load_automation(args.automate, events)) return 2;

    ysfx_config_u config{ysfx_config_new()};
    if (!config) {
        std::fprintf(stderr, "ysfx_config_new failed\n");
        return 1;
    }
    ysfx_set_log_reporter(config.get(), log_reporter);
    ysfx_guess_file_roots(config.get(), args.plugin.c_str());
    ysfx_register_builtin_audio_formats(config.get());

    ysfx_u fx{ysfx_new(config.get())};
    if (!fx) {
        std::fprintf(stderr, "ysfx_new failed\n");
        return 1;
    }
    if (!ysfx_load_file(fx.get(), args.plugin.c_str(), 0)) {
        std::fprintf(stderr, "load failed: %s\n", args.plugin.c_str());
        return 1;
    }
    // Headless: do not compile @gfx. @serialize stays available for --serialize-dump.
    if (!ysfx_compile(fx.get(), ysfx_compile_no_gfx)) {
        std::fprintf(stderr, "compile failed: %s\n", args.plugin.c_str());
        return 1;
    }

    const uint32_t n_in_pins = ysfx_get_num_inputs(fx.get());
    const uint32_t n_out_pins = ysfx_get_num_outputs(fx.get());
    if (n_out_pins < 1) {
        std::fprintf(stderr, "plugin has no outputs\n");
        return 1;
    }
    if (n_in_pins != 4 || n_out_pins < 2) {
        std::fprintf(stderr, "note: plugin declares in=%u out=%u; still feeding 4 inputs and writing 2 outputs\n", n_in_pins, n_out_pins);
    }

    ysfx_set_sample_rate(fx.get(), (ysfx_real)sr);
    ysfx_set_block_size(fx.get(), (uint32_t)args.block);

    for (const auto &sv : args.sets) apply_slider(fx.get(), sv.first, sv.second);

    const size_t frames = input.interleaved.size() / (size_t)input.channels;
    size_t ev_i = 0;
    while (ev_i < events.size() && events[ev_i].sample <= 0) {
        if (events[ev_i].sample < 0) {
            std::fprintf(stderr, "ignoring negative automation sample %lld\n", (long long)events[ev_i].sample);
        } else {
            apply_slider(fx.get(), events[ev_i].slider, events[ev_i].value);
        }
        ++ev_i;
    }

    // @init now. The first process() then runs @slider (must_compute_slider) before @block/@sample.
    ysfx_init(fx.get());

    std::vector<float> ch[4];
    for (int c = 0; c < 4; ++c) ch[c].assign(frames, 0.f);
    for (size_t i = 0; i < frames; ++i) {
        for (int c = 0; c < input.channels; ++c)
            ch[c][i] = input.interleaved[i * (size_t)input.channels + (size_t)c];
    }
    std::vector<float> outL(frames, 0.f), outR(frames, 0.f);

    ysfx_time_info_t time{};
    time.tempo = 120;
    time.playback_state = ysfx_playback_playing;
    time.time_signature[0] = 4;
    time.time_signature[1] = 4;

    const auto t0 = std::chrono::steady_clock::now();
    int64_t pos = 0;
    const int64_t n = (int64_t)frames;
    while (pos < n) {
        while (ev_i < events.size() && events[ev_i].sample < pos) ++ev_i;
        while (ev_i < events.size() && events[ev_i].sample == pos) {
            apply_slider(fx.get(), events[ev_i].slider, events[ev_i].value);
            ++ev_i;
        }
        int64_t next = std::min<int64_t>(n, pos + args.block);
        if (ev_i < events.size() && events[ev_i].sample > pos && events[ev_i].sample < next)
            next = events[ev_i].sample;
        const uint32_t count = (uint32_t)(next - pos);
        if (count == 0) break;

        time.time_position = (ysfx_real)pos / (ysfx_real)sr;
        time.beat_position = time.time_position * time.tempo / 60.0;
        ysfx_set_time_info(fx.get(), &time);

        const float *ins[4] = {ch[0].data() + pos, ch[1].data() + pos, ch[2].data() + pos, ch[3].data() + pos};
        float *outs[2] = {outL.data() + pos, outR.data() + pos};
        // Always present 4 inputs and 2 outputs. Extra inputs are ignored by a
        // plugin with fewer pins; a mono plugin's right channel is copied below.
        ysfx_process_float(fx.get(), ins, outs, 4, 2, count);
        if (n_out_pins < 2) {
            for (uint32_t i = 0; i < count; ++i) outR[(size_t)pos + i] = outL[(size_t)pos + i];
        }
        pos = next;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double audio = (double)frames / (double)sr;
    const double rtf = wall > 0 ? audio / wall : 0;

    uint64_t nan_count = 0, inf_count = 0;
    double peak = 0;
    std::vector<float> interleaved(frames * 2);
    for (size_t i = 0; i < frames; ++i) {
        const float l = outL[i];
        const float r = outR[i];
        interleaved[i * 2] = l;
        interleaved[i * 2 + 1] = r;
        for (float s : {l, r}) {
            if (std::isnan(s)) ++nan_count;
            else if (std::isinf(s)) ++inf_count;
            else peak = std::max(peak, (double)std::fabs(s));
        }
    }
    if (!write_wav_f32(args.output, sr, interleaved, 2)) return 2;

    if (args.serialize_dump) {
        ysfx_state_u state{ysfx_save_state(fx.get())};
        if (!state) {
            std::fprintf(stdout, "serialize_bytes: 0\n");
        } else {
            std::fprintf(stdout, "serialize_bytes: %zu\n", state->data_size);
            std::fprintf(stdout, "serialize_hex: ");
            if (state->data && state->data_size) hex_dump(state->data, state->data_size);
            else std::fputc('\n', stdout);
            std::fprintf(stdout, "slider_count: %u\n", state->slider_count);
            for (uint32_t i = 0; i < state->slider_count; ++i) {
                std::fprintf(stdout, "slider %u: %.17g\n", state->sliders[i].index + 1, (double)state->sliders[i].value);
            }
        }
    }

    const double peak_db = peak > 0 ? 20.0 * std::log10(peak) : -240.0;
    std::fprintf(stderr, "wall_s=%.6f\n", wall);
    std::fprintf(stderr, "rtf=%.6f\n", rtf);
    std::fprintf(stderr, "nan_count=%llu\n", (unsigned long long)nan_count);
    std::fprintf(stderr, "inf_count=%llu\n", (unsigned long long)inf_count);
    std::fprintf(stderr, "frames=%zu\n", frames);
    std::fprintf(stderr, "sample_rate=%d\n", sr);
    std::fprintf(stderr, "peak_abs=%.8g\n", peak);
    std::fprintf(stderr, "peak_dbfs=%.4f\n", peak_db);
    std::fprintf(stderr, "block=%d\n", args.block);
    return 0;
}
