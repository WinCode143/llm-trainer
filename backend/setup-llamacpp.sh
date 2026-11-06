#!/usr/bin/env bash
# Optional GGUF tooling, installed into tools/ (no sudo needed):
#   1. llama.cpp (CPU build) → convert_hf_to_gguf.py + llama-quantize, to export models to GGUF
#   2. llama-cpp-python built with Vulkan → GGUF inference on the GPU without the CUDA toolkit
#      (needs SPIRV-Headers, which are built locally in tools/spirv-headers)
# llama.cpp is cloned here on demand instead of being vendored or added as a submodule: it's ~300 MB,
# only the GGUF features need it, and this way you always get a current version.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$ROOT/tools"
DIR="$TOOLS/llama.cpp"
mkdir -p "$TOOLS"

if [ -d "$DIR/.git" ]; then
  echo "[1/4] Updating llama.cpp…"; git -C "$DIR" pull --ff-only -q
else
  echo "[1/4] Cloning llama.cpp…"; git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$DIR"
fi

echo "[2/4] Building llama-quantize…"
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF >/dev/null
cmake --build "$DIR/build" --target llama-quantize -j "$(nproc)"

if ! command -v glslc >/dev/null || [ ! -f /usr/include/vulkan/vulkan.h ]; then
  echo "[!] Vulkan headers/shaderc not found: GGUF export works, but GGUF inference will run on the CPU only."
  echo "    For the GPU install them first: sudo apt install libvulkan-dev glslc   (Arch: sudo pacman -S vulkan-headers shaderc)"
  export CMAKE_ARGS="-DGGML_NATIVE=on"
else
  if [ ! -f "$TOOLS/spirv-headers/include/spirv/unified1/spirv.hpp" ]; then
    echo "[3/4] Building SPIRV-Headers…"
    rm -rf "$TOOLS/SPIRV-Headers-src"
    git clone -q --depth 1 https://github.com/KhronosGroup/SPIRV-Headers "$TOOLS/SPIRV-Headers-src"
    cmake -S "$TOOLS/SPIRV-Headers-src" -B "$TOOLS/SPIRV-Headers-src/build" -DCMAKE_INSTALL_PREFIX="$TOOLS/spirv-headers" \
      -DSPIRV_HEADERS_ENABLE_TESTS=OFF >/dev/null
    cmake --install "$TOOLS/SPIRV-Headers-src/build" >/dev/null
    rm -rf "$TOOLS/SPIRV-Headers-src"
  else
    echo "[3/4] SPIRV-Headers already installed"
  fi
  export CMAKE_PREFIX_PATH="$TOOLS/spirv-headers" CPLUS_INCLUDE_PATH="$TOOLS/spirv-headers/include"
fi

echo "[4/4] Building llama-cpp-python (takes a few minutes)…"
cd "$ROOT/backend"
uv sync --extra gguf --reinstall-package llama-cpp-python
.venv/bin/python -c "import llama_cpp; print('[ok] llama-cpp-python', llama_cpp.__version__, 'GPU:', llama_cpp.llama_supports_gpu_offload())"
echo "[ok] Done: $DIR"
