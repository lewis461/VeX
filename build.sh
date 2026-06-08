#!/bin/bash
###############################################################################
# VeX build script.
#
# Integrates the VeX plugin into a FAISS checkout, builds libfaiss with the
# plugin, and compiles the two drivers (build_index, search_index).
#
# Host-only by default. For the DPU/pipelined path, set WITH_DOCA=1 (requires
# NVIDIA DOCA + a BlueField DPU).
#
# Env:
#   FAISS_SRC   path to a FAISS source checkout (default: ./faiss; cloned if absent)
#   WITH_DOCA   1 to enable the DOCA/DPU path (default: 0)
#   JOBS        parallel build jobs (default: nproc)
###############################################################################
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAISS_SRC="${FAISS_SRC:-$REPO/faiss}"
WITH_DOCA="${WITH_DOCA:-0}"
JOBS="${JOBS:-$(nproc)}"
FAISS_TAG="${FAISS_TAG:-v1.12.0}"

# 1) Obtain FAISS
if [ ! -d "$FAISS_SRC" ]; then
  echo ">> Cloning FAISS ($FAISS_TAG) into $FAISS_SRC"
  git clone --depth 1 --branch "$FAISS_TAG" https://github.com/facebookresearch/faiss "$FAISS_SRC"
fi

# 2) Integrate the plugin into the FAISS tree
echo ">> Installing VeX plugin into $FAISS_SRC/plugin"
rm -rf "$FAISS_SRC/plugin"
cp -r "$REPO/plugin" "$FAISS_SRC/plugin"
if ! grep -q "add_subdirectory(plugin)" "$FAISS_SRC/CMakeLists.txt"; then
  echo "add_subdirectory(plugin)" >> "$FAISS_SRC/CMakeLists.txt"
fi

# 3) Configure + build libfaiss (host-only unless WITH_DOCA=1)
DOCA_FLAG="OFF"; [ "$WITH_DOCA" = "1" ] && DOCA_FLAG="ON"
echo ">> Building libfaiss (DOCA=$DOCA_FLAG)"
cmake -S "$FAISS_SRC" -B "$FAISS_SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_WITH_DOCA=$DOCA_FLAG -DBUILD_TESTING=OFF
cmake --build "$FAISS_SRC/build" --target faiss -j"$JOBS"

# 4) Compile the drivers
LIB="$FAISS_SRC/build/faiss"
mkdir -p "$REPO/bin"
CXXFLAGS="-std=c++17 -O2 -fopenmp -Wno-deprecated-declarations -I$FAISS_SRC -I$REPO/plugin"
LDFLAGS="-L$LIB -lfaiss -lblas -llapack -lpthread"
DEFS=""; [ "$WITH_DOCA" = "1" ] && DEFS="-DFAISS_WITH_DOCA"
echo ">> Compiling drivers"
g++ $CXXFLAGS $DEFS "$REPO/scripts/build_index.cpp"  -o "$REPO/bin/build_index"  $LDFLAGS
g++ $CXXFLAGS $DEFS "$REPO/scripts/search_index.cpp" -o "$REPO/bin/search_index" $LDFLAGS

echo ""
echo "Done. Binaries in $REPO/bin/"
echo "  build_index  -- build a VeX index"
echo "  search_index -- search & evaluate recall"
