#!/usr/bin/env bash
set -euo pipefail

# This script bootstraps a local build and installation of the QUGaR library
# and its Python interface following the upstream installation guide:
# https://pantolin.github.io/qugar/main/installation.html
#
# It will:
#   1. clone or update the QUGaR repository inside .deps/qugar
#   2. check if liblapacke is available and enable it if found
#   3. build the C++ core in Release mode into .deps/qugar/build
#   4. install the C++ artifacts under .deps/qugar/install
#   5. install the Python interface into the current Python environment
#   6. remove the temporary build dependencies folder
#
# Prerequisites:
#   - C++20 compiler toolchain (e.g. clang >= 14 or gcc >= 10)
#   - CMake >= 3.20
#   - (optional) Ninja for faster builds; falls back to Unix Makefiles otherwise
#   - Python 3.10+ with pip
#   - git
#   - (optional) liblapacke development library for performance
#
# --- Recommended Setup ---
# It is highly recommended to perform the installation within a dedicated conda
# environment.
#
# 1. Create and activate a new conda environment:
#    conda create -n qugar-env python=3.10
#    conda activate qugar-env
#
# 2. Install FEniCSx (dolfinx) into this environment:
#    conda install -c conda-forge fenics-dolfinx
#    (See https://github.com/FEniCS/dolfinx?tab=readme-ov-file#conda for details)
#
# 3. Install liblapacke for better performance:
#    conda install -c conda-forge liblapacke libtmglib
# -------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${REPO_ROOT}/.deps"
QUGAR_DIR="${DEPS_DIR}/qugar"
BUILD_DIR="${QUGAR_DIR}/build"
INSTALL_DIR="${QUGAR_DIR}/install"

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_CMD=("${PYTHON_BIN}" "-m" "pip")

JOBS="${JOBS:-$(command -v sysctl &> /dev/null && sysctl -n hw.ncpu || nproc || echo 4)}"
GENERATOR="Ninja"

# --- Function to check for a working liblapacke installation ---
has_lapacke() {
    local check_dir
    check_dir=$(mktemp -d 2>/dev/null || mktemp -d -t 'check_lapacke')
    trap 'rm -rf -- "$check_dir"' RETURN

    cat > "${check_dir}/check.cpp" <<EOF
#include <lapacke.h>
int main() {
    // We don't need to run it, just check if it links.
    // Calling a function prevents the linker from optimizing the library away.
    (void)LAPACKE_dgesv;
    return 0;
}
EOF

    local CXX_COMPILER="${CXX:-g++}"
    if ! command -v "${CXX_COMPILER}" &> /dev/null; then
        CXX_COMPILER="c++"
        if ! command -v "${CXX_COMPILER}" &> /dev/null; then
            echo "[WARN] No C++ compiler found to check for LAPACKE. Assuming not present." >&2
            return 1
        fi
    fi

    local CXX_FLAGS=""
    local LDFLAGS=""
    # If we are in a conda environment, add its paths
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        CXX_FLAGS="-I${CONDA_PREFIX}/include"
        LDFLAGS="-L${CONDA_PREFIX}/lib"
    fi

    # Attempt to compile and link against liblapacke.
    if "${CXX_COMPILER}" "${check_dir}/check.cpp" -o "${check_dir}/check.out" ${CXX_FLAGS} ${LDFLAGS} -llapacke &> /dev/null; then
        return 0 # Success
    else
        return 1 # Failure
    fi
}


if ! command -v cmake &> /dev/null; then
  echo "[ERROR] cmake not found. Please install CMake >= 3.20." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" &> /dev/null; then
  echo "[ERROR] ${PYTHON_BIN} not found." >&2
  exit 1
fi

if ! command -v git &> /dev/null; then
  echo "[ERROR] git not found." >&2
  exit 1
fi

if ! command -v ninja &> /dev/null; then
  echo "[WARN] Ninja not found. Falling back to Unix Makefiles generator." >&2
  GENERATOR="Unix Makefiles"
fi

echo "[INFO] Using generator: ${GENERATOR}"

echo "[INFO] Preparing dependencies directory at ${DEPS_DIR}"
mkdir -p "${DEPS_DIR}"

if [ ! -d "${QUGAR_DIR}" ]; then
  echo "[INFO] Cloning QUGaR repository..."
  git clone https://github.com/pantolin/qugar.git "${QUGAR_DIR}"
else
  echo "[INFO] Updating existing QUGaR clone..."
  git -C "${QUGAR_DIR}" fetch --all
  git -C "${QUGAR_DIR}" reset --hard origin/main
fi

TPMS_HEADER="${QUGAR_DIR}/cpp/include/qugar/tpms_lib.hpp"
if [ -f "${TPMS_HEADER}" ]; then
  echo "[INFO] Ensuring tpms_lib.hpp has dependent-type qualifiers..."
  "${PYTHON_BIN}" - "$TPMS_HEADER" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
original = path.read_text()

patched = original.replace(
    "template<typename T> using Gradient = qugar::impl::ImplicitFunc<dim>::template Gradient<T>;",
    "template<typename T> using Gradient = typename qugar::impl::ImplicitFunc<dim>::template Gradient<T>;"
)
patched = patched.replace(
    "template<typename T> using Hessian = qugar::impl::ImplicitFunc<dim>::template Hessian<T>;",
    "template<typename T> using Hessian = typename qugar::impl::ImplicitFunc<dim>::template Hessian<T>;"
)

if patched != original:
    path.write_text(patched)
    print("[INFO]   Applied typename fix to tpms aliases.")
else:
    print("[INFO]   tpms aliases already patched.")
PY
fi

# Check for liblapacke and prepare CMake arguments
CMAKE_EXTRA_ARGS=()
if has_lapacke; then
  echo "[INFO] Found liblapacke. Configuring QUGaR with LAPACKE support."
  CMAKE_EXTRA_ARGS+=("-DQUGAR_WITH_LAPACKE=ON")
else
  echo "[INFO] liblapacke not found or not usable. Building QUGaR without LAPACKE support."
fi

echo "[INFO] Configuring C++ core build..."
cmake \
  -G "${GENERATOR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  "${CMAKE_EXTRA_ARGS[@]}" \
  -B "${BUILD_DIR}" \
  -S "${QUGAR_DIR}/cpp"

echo "[INFO] Building C++ core with ${JOBS} parallel jobs..."
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo "[INFO] Installing C++ artifacts to ${INSTALL_DIR}"
cmake --install "${BUILD_DIR}"

export CMAKE_PREFIX_PATH="${INSTALL_DIR}:${CMAKE_PREFIX_PATH:-}"

PYTHON_DIR="${QUGAR_DIR}/python"

if [ ! -f "${PYTHON_DIR}/build-requirements.txt" ]; then
  echo "[ERROR] Missing build requirements file at ${PYTHON_DIR}/build-requirements.txt" >&2
  exit 1
fi

echo "[INFO] Installing Python build requirements..."
"${PIP_CMD[@]}" -v install -r "${PYTHON_DIR}/build-requirements.txt"

echo "[INFO] Installing QUGaR Python interface..."
"${PIP_CMD[@]}" -v install --no-build-isolation "${PYTHON_DIR}" -U

echo "[SUCCESS] QUGaR installation complete."

echo "[INFO] Cleaning up build dependencies..."
cd "${REPO_ROOT}"
rm -rf "${DEPS_DIR}"

echo "[SUCCESS] QUGaR installation and cleanup complete."
