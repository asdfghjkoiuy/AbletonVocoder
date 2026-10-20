#!/usr/bin/env bash
# Clone ysfx (if needed), build tools/build/jsfx_render, and create tools/.venv.
# Re-running is safe. Network is required only for the first clone / pip install.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
YSFX_DIR="$ROOT/tools/ysfx"
BUILD_DIR="$ROOT/tools/build"
VENV="$ROOT/tools/.venv"
PY="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"

# JoepVanlier/ysfx @ 5c3452fee62583aa3d1b7e877d0c758c4024af89
# ("ci: allow manual code signing on older macs", version 0.0.47)
YSFX_URL="https://github.com/JoepVanlier/ysfx.git"
YSFX_COMMIT="5c3452fee62583aa3d1b7e877d0c758c4024af89"
YSFX_FALLBACK_URL="https://github.com/jpcima/ysfx.git"

if [[ ! -d "$YSFX_DIR/.git" ]]; then
  echo "cloning ysfx..."
  if ! git clone "$YSFX_URL" "$YSFX_DIR"; then
    echo "primary clone failed, trying ${YSFX_FALLBACK_URL}"
    git clone "$YSFX_FALLBACK_URL" "$YSFX_DIR"
  fi
fi

if git -C "$YSFX_DIR" cat-file -e "${YSFX_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$YSFX_DIR" checkout --detach "$YSFX_COMMIT"
else
  echo "fetching pinned ysfx commit ${YSFX_COMMIT}"
  if ! git -C "$YSFX_DIR" fetch --depth 1 origin "$YSFX_COMMIT"; then
    git -C "$YSFX_DIR" fetch origin
  fi
  git -C "$YSFX_DIR" checkout --detach "$YSFX_COMMIT"
fi

echo "ysfx commit: $(git -C "$YSFX_DIR" rev-parse HEAD)"

# dr_wav / dr_flac are a submodule. The plugin and gfx submodules are not required.
git -C "$YSFX_DIR" submodule update --init --depth 1 thirdparty/dr_libs

# Upstream CMake adds an x64 assembly unit whenever the portable interpreter is
# off. On arm64 that unit cannot assemble, and it is also unnecessary: WDL's
# aarch64 JIT is compiled straight into nseel-compiler.c.
python3 - "$YSFX_DIR/cmake.wdl.txt" << 'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
old = '''    if(NOT MSVC)
        target_sources(eel2 PRIVATE "sources/eel2-gas/sources/asm-nseel-x64-sse.S")'''
new = '''    if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        target_sources(eel2 PRIVATE "sources/eel2-gas/sources/asm-nseel-x64-sse.S")'''
if old in text:
    path.write_text(text.replace(old, new, 1))
elif new not in text:
    sys.exit("ysfx cmake.wdl.txt changed; cannot select the arm64 EEL JIT")
PY

# v1 reads spl2/spl3 without declaring pins. ysfx otherwise only copies the
# declared pin count (default stereo), so the sidechain never arrives.
python3 - "$YSFX_DIR/sources/ysfx.cpp" << 'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
old = '''        const uint32_t num_code_ins = (uint32_t)fx->source.main->header.in_pins.size();
        const uint32_t num_code_outs = (uint32_t)fx->source.main->header.out_pins.size();
        if (num_ins > num_code_ins)
            num_ins = num_code_ins;
'''
new = '''        const uint32_t num_code_ins = (uint32_t)fx->source.main->header.in_pins.size();
        const uint32_t num_code_outs = (uint32_t)fx->source.main->header.out_pins.size();
        // Effects that never declare pins default to stereo, but some (the v1
        // vocoder) still read spl2/spl3 as a sidechain. Publish every buffer
        // the host actually passed, up to the spl array.
        const uint32_t num_spl_ins = std::min(orig_num_ins, (uint32_t)ysfx_max_channels);
        if (num_ins > num_code_ins)
            num_ins = num_code_ins;
'''
old_loop = '''                for (uint32_t ch = 0; ch < num_ins; ++ch)
                    *spl[ch] = (EEL_F)ins[ch][i] + denorm_value;
                for (uint32_t ch = num_ins; ch < num_code_ins; ++ch)
                    *spl[ch] = denorm_value;
'''
new_loop = '''                for (uint32_t ch = 0; ch < num_spl_ins; ++ch)
                    *spl[ch] = (EEL_F)ins[ch][i] + denorm_value;
                for (uint32_t ch = num_spl_ins; ch < num_code_ins; ++ch)
                    *spl[ch] = denorm_value;
'''
if "num_spl_ins" not in text:
    if old not in text or old_loop not in text:
        sys.exit("ysfx.cpp layout changed; cannot feed undeclared sidechain channels")
    text = text.replace(old, new, 1).replace(old_loop, new_loop, 1)
    path.write_text(text)
PY

if [[ ! -x "$PY" ]]; then
  echo "python3.10 not found at $PY" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PY" -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c "import numpy, scipy, soundfile" >/dev/null 2>&1; then
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r "$ROOT/tools/requirements.txt"
fi

cmake -S "$ROOT/tools" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release -j

echo "built: $BUILD_DIR/jsfx_render"
echo "python: $VENV/bin/python"
