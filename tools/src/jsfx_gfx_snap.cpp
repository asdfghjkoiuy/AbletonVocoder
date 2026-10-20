// Headless @gfx snapshotter.
//
//   jsfx_gfx_snap <plugin.jsfx> <in.wav> --png out.png [--size WxH]
//       [--scale S] [--seconds T] [--sr N] [--block N]
//       [--set IDX=VAL ...] [--mouse script.txt] [--warmup N]
//
// Processes the input (4 channels presented to the plugin; missing carrier
// channels are zero) so DISP_ENV has content, then runs @gfx into a BGRA
// framebuffer via ysfx/LICE and writes a PNG.
//
// --size is the framebuffer in pixels (default 780x290). --scale is the
// hidpi factor passed to ysfx (1, or 2 for Retina). Mouse-script coordinates
// are framebuffer pixels. A positive wheel step is one REAPER notch.
//
// Mouse script (one command per line, # comments):
//   move x y
//   down [x y]
//   up [x y]
//   wheel steps [x y]
//   mods none|shift|alt|ctrl|super     (combine with +)
//   frame n
// Each command except mods and frame runs one @gfx frame. frame n runs n
// frames with the mouse left as it is. mods sticks until the next mods line.

#include "ysfx.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma clang diagnostic pop

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Wav {
    int sr = 0;
    int channels = 0;
    std::vector<float> interleaved;
};

uint16_t rd_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

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
    int format = 0, channels = 0, sr = 0, bits = 0;
    const uint8_t *data = nullptr;
    size_t data_bytes = 0;
    size_t pos = 12;
    while (pos + 8 <= file.size()) {
        const uint8_t *chunk = file.data() + pos;
        const uint32_t sz = rd_u32(chunk + 4);
        const size_t body = pos + 8;
        if (body + sz > file.size()) return false;
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (sz < 16) return false;
            format = rd_u16(file.data() + body);
            channels = rd_u16(file.data() + body + 2);
            sr = (int)rd_u32(file.data() + body + 4);
            bits = rd_u16(file.data() + body + 14);
            if (format == 0xFFFE) {
                if (sz < 40) return false;
                format = rd_u16(file.data() + body + 24);
            }
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data = file.data() + body;
            data_bytes = sz;
        }
        pos = body + sz + (sz & 1u);
    }
    if (!data || channels < 1 || channels > 4 || sr < 1000 || bits < 8) {
        std::fprintf(stderr, "wav needs 1, 2 or 4 channels and a sample rate\n");
        return false;
    }
    const int bytes_per = bits / 8;
    const size_t frames = data_bytes / (size_t)(bytes_per * channels);
    if (frames == 0) return false;
    wav.sr = sr;
    wav.channels = channels;
    wav.interleaved.assign(frames * (size_t)channels, 0.f);
    const uint8_t *p = data;
    for (size_t i = 0; i < frames * (size_t)channels; ++i) {
        float s = 0.f;
        if (format == 1 && bits == 16) {
            int16_t v = (int16_t)rd_u16(p);
            s = (float)v / 32768.f;
        } else if (format == 3 && bits == 32) {
            uint32_t u = rd_u32(p);
            std::memcpy(&s, &u, 4);
        } else if (format == 1 && bits == 24) {
            int32_t v = (int32_t)(p[0] | (p[1] << 8) | (p[2] << 16));
            if (v & 0x800000) v |= ~0xFFFFFF;
            s = (float)v / 8388608.f;
        } else if (format == 1 && bits == 32) {
            int32_t v = (int32_t)rd_u32(p);
            s = (float)((double)v / 2147483648.0);
        } else {
            std::fprintf(stderr, "unsupported wav format %d bits %d\n", format, bits);
            return false;
        }
        wav.interleaved[i] = s;
        p += bytes_per;
    }
    return true;
}

struct Args {
    std::string plugin;
    std::string input;
    std::string png;
    std::string mouse;
    int width = 780;
    int height = 290;
    double scale = 1;
    double seconds = 1.5;
    int sr = 0;
    int block = 512;
    int warmup = 2;
    std::vector<std::pair<int, double>> sets;
};

void usage() {
    std::fprintf(stderr,
                 "Usage: jsfx_gfx_snap <plugin.jsfx> <in.wav> --png out.png\n"
                 "  [--size WxH] [--scale S] [--seconds T] [--sr N] [--block N]\n"
                 "  [--set IDX=VAL ...] [--mouse script.txt] [--warmup N]\n");
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

bool parse_size(const std::string &text, int &w, int &h) {
    const auto x = text.find('x');
    if (x == std::string::npos || x == 0) return false;
    try {
        w = std::stoi(text.substr(0, x));
        h = std::stoi(text.substr(x + 1));
    } catch (...) {
        return false;
    }
    return w >= 16 && w <= 8192 && h >= 16 && h <= 8192;
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
        if (a == "--png") {
            const char *v = need("--png");
            if (!v) return false;
            args.png = v;
        } else if (a == "--size") {
            const char *v = need("--size");
            if (!v) return false;
            if (!parse_size(v, args.width, args.height)) {
                std::fprintf(stderr, "bad --size '%s'\n", v);
                return false;
            }
        } else if (a == "--scale") {
            const char *v = need("--scale");
            if (!v) return false;
            args.scale = std::atof(v);
            if (!(args.scale >= 1.0 && args.scale <= 4.0)) {
                std::fprintf(stderr, "bad --scale\n");
                return false;
            }
        } else if (a == "--seconds") {
            const char *v = need("--seconds");
            if (!v) return false;
            args.seconds = std::atof(v);
            if (!(args.seconds > 0.0 && args.seconds < 3600.0)) return false;
        } else if (a == "--sr") {
            const char *v = need("--sr");
            if (!v) return false;
            args.sr = std::atoi(v);
        } else if (a == "--block") {
            const char *v = need("--block");
            if (!v) return false;
            args.block = std::atoi(v);
            if (args.block < 1) return false;
        } else if (a == "--warmup") {
            const char *v = need("--warmup");
            if (!v) return false;
            args.warmup = std::atoi(v);
            if (args.warmup < 0) return false;
        } else if (a == "--set") {
            const char *v = need("--set");
            if (!v) return false;
            int idx = 0;
            double value = 0;
            if (!parse_set(v, idx, value)) {
                std::fprintf(stderr, "bad --set '%s'\n", v);
                return false;
            }
            args.sets.emplace_back(idx, value);
        } else if (a == "--mouse") {
            const char *v = need("--mouse");
            if (!v) return false;
            args.mouse = v;
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
    if (pos.size() != 2 || args.png.empty()) {
        usage();
        return false;
    }
    args.plugin = pos[0];
    args.input = pos[1];
    return true;
}

enum class Act { Move, Down, Up, Wheel, Mods, Frame };

struct ActCmd {
    Act act = Act::Move;
    double x = 0;
    double y = 0;
    double v = 0;
    bool has_xy = false;
    uint32_t mods = 0;
    int line = 0;
};

bool parse_mods(const std::string &text, uint32_t &mods) {
    mods = 0;
    if (text == "none" || text.empty()) return true;
    std::string cur;
    for (size_t i = 0; i <= text.size(); ++i) {
        const char c = i == text.size() ? '+' : text[i];
        if (c == '+' || c == ',' || c == '|') {
            if (cur == "shift") mods |= ysfx_mod_shift;
            else if (cur == "alt" || cur == "option" || cur == "opt") mods |= ysfx_mod_alt;
            else if (cur == "ctrl" || cur == "control") mods |= ysfx_mod_ctrl;
            else if (cur == "super" || cur == "cmd" || cur == "command") mods |= ysfx_mod_super;
            else if (cur == "none" || cur.empty()) {
            } else {
                return false;
            }
            cur.clear();
        } else {
            cur.push_back((char)std::tolower((unsigned char)c));
        }
    }
    return true;
}

bool load_mouse(const std::string &path, std::vector<ActCmd> &cmds) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open mouse script %s\n", path.c_str());
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
        std::stringstream ss(line.substr(b));
        std::string op;
        ss >> op;
        ActCmd cmd;
        cmd.line = lineno;
        if (op == "move") {
            cmd.act = Act::Move;
            if (!(ss >> cmd.x >> cmd.y)) {
                std::fprintf(stderr, "mouse line %d: move x y\n", lineno);
                return false;
            }
            cmd.has_xy = true;
        } else if (op == "down" || op == "up") {
            cmd.act = op == "down" ? Act::Down : Act::Up;
            if (ss >> cmd.x) {
                if (!(ss >> cmd.y)) {
                    std::fprintf(stderr, "mouse line %d: %s [x y]\n", lineno, op.c_str());
                    return false;
                }
                cmd.has_xy = true;
            }
        } else if (op == "wheel") {
            cmd.act = Act::Wheel;
            if (!(ss >> cmd.v)) {
                std::fprintf(stderr, "mouse line %d: wheel steps [x y]\n", lineno);
                return false;
            }
            if (ss >> cmd.x) {
                if (!(ss >> cmd.y)) {
                    std::fprintf(stderr, "mouse line %d: wheel steps [x y]\n", lineno);
                    return false;
                }
                cmd.has_xy = true;
            }
        } else if (op == "mods") {
            cmd.act = Act::Mods;
            std::string spec;
            ss >> spec;
            if (!parse_mods(spec, cmd.mods)) {
                std::fprintf(stderr, "mouse line %d: bad mods '%s'\n", lineno, spec.c_str());
                return false;
            }
        } else if (op == "frame") {
            cmd.act = Act::Frame;
            if (!(ss >> cmd.v) || cmd.v < 0 || cmd.v > 100000) {
                std::fprintf(stderr, "mouse line %d: frame n\n", lineno);
                return false;
            }
        } else {
            std::fprintf(stderr, "mouse line %d: unknown '%s'\n", lineno, op.c_str());
            return false;
        }
        cmds.push_back(cmd);
    }
    return true;
}

void log_reporter(intptr_t, ysfx_log_level level, const char *message) {
    std::fprintf(stderr, "ysfx %s: %s\n", ysfx_log_level_string(level), message);
}

bool write_png(const std::string &path, const uint8_t *bgra, int w, int h) {
    std::vector<uint8_t> rgba((size_t)w * (size_t)h * 4u);
    for (int i = 0; i < w * h; ++i) {
        rgba[(size_t)i * 4 + 0] = bgra[(size_t)i * 4 + 2];
        rgba[(size_t)i * 4 + 1] = bgra[(size_t)i * 4 + 1];
        rgba[(size_t)i * 4 + 2] = bgra[(size_t)i * 4 + 0];
        rgba[(size_t)i * 4 + 3] = 255;
    }
    std::filesystem::path out(path);
    if (out.has_parent_path() && !out.parent_path().empty())
        std::filesystem::create_directories(out.parent_path());
    if (!stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4)) {
        std::fprintf(stderr, "png write failed: %s\n", path.c_str());
        return false;
    }
    return true;
}

void present_mouse(ysfx_t *fx, int x, int y, bool down, uint32_t mods, double wheel_steps) {
    const uint32_t buttons = down ? (uint32_t)ysfx_button_left : 0u;
    // ysfx adds 512*wheel into mouse_wheel. One REAPER notch is 120.
    const ysfx_real wheel = (ysfx_real)(wheel_steps * (120.0 / 512.0));
    ysfx_gfx_update_mouse(fx, mods, x, y, buttons, wheel, 0);
}

void run_frame(ysfx_t *fx, int x, int y, bool down, uint32_t mods, double wheel_steps) {
    present_mouse(fx, x, y, down, mods, wheel_steps);
    ysfx_gfx_run(fx);
}

}  // namespace

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    std::vector<ActCmd> mouse;
    if (!args.mouse.empty() && !load_mouse(args.mouse, mouse)) return 2;

    Wav input;
    if (!read_wav(args.input, input)) return 2;
    const int sr = args.sr > 0 ? args.sr : input.sr;

    ysfx_config_u config{ysfx_config_new()};
    ysfx_set_log_reporter(config.get(), log_reporter);
    ysfx_guess_file_roots(config.get(), args.plugin.c_str());
    ysfx_register_builtin_audio_formats(config.get());
    ysfx_u fx{ysfx_new(config.get())};
    if (!fx || !ysfx_load_file(fx.get(), args.plugin.c_str(), 0)) {
        std::fprintf(stderr, "load failed: %s\n", args.plugin.c_str());
        return 1;
    }
    if (!ysfx_compile(fx.get(), 0)) {
        std::fprintf(stderr, "compile failed: %s\n", args.plugin.c_str());
        return 1;
    }
    if (!ysfx_has_section(fx.get(), ysfx_section_gfx)) {
        std::fprintf(stderr, "plugin has no @gfx section\n");
        return 1;
    }

    ysfx_set_sample_rate(fx.get(), (ysfx_real)sr);
    ysfx_set_block_size(fx.get(), (uint32_t)args.block);
    for (const auto &sv : args.sets)
        ysfx_slider_set_value(fx.get(), (uint32_t)(sv.first - 1), (ysfx_real)sv.second, true);
    ysfx_init(fx.get());

    const size_t in_frames = input.interleaved.size() / (size_t)input.channels;
    size_t frames = in_frames;
    const size_t want = (size_t)std::llround(args.seconds * (double)sr);
    if (want < frames) frames = want;
    std::vector<float> ch[4];
    for (int c = 0; c < 4; ++c) ch[c].assign(frames, 0.f);
    const double ratio = (double)input.sr / (double)sr;
    for (size_t i = 0; i < frames; ++i) {
        const double src = (double)i * ratio;
        size_t i0 = (size_t)src;
        if (i0 >= in_frames) i0 = in_frames - 1;
        const size_t i1 = std::min(i0 + 1, in_frames - 1);
        const float f = (input.sr == sr) ? 0.f : (float)(src - (double)i0);
        for (int c = 0; c < input.channels; ++c) {
            const float a = input.interleaved[i0 * (size_t)input.channels + (size_t)c];
            const float b = input.interleaved[i1 * (size_t)input.channels + (size_t)c];
            ch[c][i] = a + (b - a) * f;
        }
        // A mono file is the modulator. Copy it to the right channel so the
        // stereo analysis envelope is not half-silent.
        if (input.channels == 1) ch[1][i] = ch[0][i];
    }
    std::vector<float> outL(frames, 0.f), outR(frames, 0.f);
    ysfx_time_info_t time{};
    time.tempo = 120;
    time.playback_state = ysfx_playback_playing;
    time.time_signature[0] = 4;
    time.time_signature[1] = 4;
    int64_t pos = 0;
    while (pos < (int64_t)frames) {
        const uint32_t count = (uint32_t)std::min<int64_t>(args.block, (int64_t)frames - pos);
        time.time_position = (ysfx_real)pos / (ysfx_real)sr;
        time.beat_position = time.time_position * time.tempo / 60.0;
        ysfx_set_time_info(fx.get(), &time);
        const float *ins[4] = {ch[0].data() + pos, ch[1].data() + pos, ch[2].data() + pos, ch[3].data() + pos};
        float *outs[2] = {outL.data() + pos, outR.data() + pos};
        ysfx_process_float(fx.get(), ins, outs, 4, 2, count);
        pos += count;
    }
    std::fprintf(stderr, "audio_frames=%zu sr=%d\n", frames, sr);

    const int w = args.width;
    const int h = args.height;
    std::vector<uint8_t> pixels((size_t)w * (size_t)h * 4u, 0);
    ysfx_gfx_config_t gc{};
    gc.user_data = nullptr;
    gc.pixel_width = (uint32_t)w;
    gc.pixel_height = (uint32_t)h;
    gc.pixel_stride = (uint32_t)w * 4u;
    gc.pixels = pixels.data();
    gc.scale_factor = (ysfx_real)args.scale;
    gc.show_menu = nullptr;
    gc.set_cursor = nullptr;
    gc.get_drop_file = nullptr;
    ysfx_gfx_setup(fx.get(), &gc);
    ysfx_gfx_set_window_state(fx.get(), true, true, true);

    int mx = 0, my = 0;
    bool down = false;
    uint32_t mods = 0;
    for (int i = 0; i < args.warmup; ++i) run_frame(fx.get(), mx, my, down, mods, 0);

    for (const ActCmd &cmd : mouse) {
        if (cmd.act == Act::Mods) {
            mods = cmd.mods;
            continue;
        }
        if (cmd.act == Act::Frame) {
            const int n = (int)cmd.v;
            for (int i = 0; i < n; ++i) run_frame(fx.get(), mx, my, down, mods, 0);
            continue;
        }
        if (cmd.has_xy) {
            mx = (int)std::lround(cmd.x);
            my = (int)std::lround(cmd.y);
        }
        double wheel = 0;
        if (cmd.act == Act::Down) down = true;
        else if (cmd.act == Act::Up) down = false;
        else if (cmd.act == Act::Wheel) wheel = cmd.v;
        run_frame(fx.get(), mx, my, down, mods, wheel);
    }

    if (!write_png(args.png, pixels.data(), w, h)) return 1;

    const bool undo = ysfx_fetch_want_undopoint(fx.get());
    const uint64_t automate = ysfx_fetch_slider_automations(fx.get(), 0);
    std::printf("png: %s\n", std::filesystem::absolute(args.png).string().c_str());
    std::printf("size: %dx%d\n", w, h);
    std::printf("scale: %.3f\n", args.scale);
    std::printf("wants_retina: %d\n", ysfx_gfx_wants_retina(fx.get()) ? 1 : 0);
    if (ysfx_real *rw = ysfx_find_var(fx.get(), "gfx_w"))
        std::printf("gfx_w: %.3f\n", (double)*rw);
    if (ysfx_real *rh = ysfx_find_var(fx.get(), "gfx_h"))
        std::printf("gfx_h: %.3f\n", (double)*rh);
    if (ysfx_real *rr = ysfx_find_var(fx.get(), "gfx_ext_retina"))
        std::printf("gfx_ext_retina: %.3f\n", (double)*rr);
    if (ysfx_real *gs = ysfx_find_var(fx.get(), "gui_s"))
        std::printf("gui_s: %.4f\n", (double)*gs);
    std::printf("undo_point: %d\n", undo ? 1 : 0);
    std::printf("automate_mask: 0x%llx\n", (unsigned long long)automate);
    const uint32_t nsl = 64;
    for (uint32_t i = 0; i < nsl; ++i) {
        if (!ysfx_slider_exists(fx.get(), i)) continue;
        const char *name = ysfx_slider_get_name(fx.get(), i);
        std::printf("slider %u %s: %.10g\n", i + 1, name ? name : "", (double)ysfx_slider_get_value(fx.get(), i));
    }
    ysfx_real att[40];
    ysfx_read_vmem(fx.get(), 0, att, 40);
    std::printf("band_att:");
    for (int i = 0; i < 40; ++i) std::printf(" %.4g", (double)att[i]);
    std::printf("\n");
    return 0;
}
