#!/bin/bash
# =============================================================================
# cuTile Hopper Hack - Run cuTile & hw1-asr on non-Blackwell GPUs
# =============================================================================
#
# This script allows you to run cuTile tutorials AND hw1-asr on older GPUs
# (Ada Lovelace, Ampere, etc.) by injecting a compatibility layer that
# translates cuTile API calls to CuPy RawKernel.
#
# Usage:
#   ./hack.sh <python_script.py>
#   ./hack.sh 1-vectoradd/vectoradd.py
#   ./hack.sh 7-attention/attention.py
#
# Or source it to enable the hack in current shell (RECOMMENDED):
#   source cutile-tutorial/hack.sh   # from project root
#   python cutile-tutorial/1-vectoradd/vectoradd.py
#   python hw1-asr/benchmark_student.py glm_asr_scratch
#
# =============================================================================

# Get absolute path of this script (works even when sourced)
_HACK_SH_SOURCE="${BASH_SOURCE[0]:-$0}"
_HACK_SH_DIR="$(dirname "${_HACK_SH_SOURCE}")"
SCRIPT_DIR="$(cd "${_HACK_SH_DIR}" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
HACK_DIR="${SCRIPT_DIR}/hack-hopper"
unset _HACK_SH_SOURCE _HACK_SH_DIR

# Set CUDA environment variables
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# CuPy needs CUDA include path for compilation
export CFLAGS="-I${CUDA_HOME}/include ${CFLAGS}"
export CXXFLAGS="-I${CUDA_HOME}/include ${CXXFLAGS}"
export CUPY_CUDA_PATH="${CUDA_HOME}"

# Inject the compatibility layer by prepending to PYTHONPATH
export PYTHONPATH="${HACK_DIR}:${PYTHONPATH}"

# Export project root for hw1-asr to find resources
export MLS_PROJECT_ROOT="${PROJECT_ROOT}"

# Function to check GPU compatibility
check_gpu() {
    python3 -c "
import cupy as cp
cc = cp.cuda.Device().compute_capability
major = int(cc[:-1])
if major >= 10:
    print('[cuTile] Blackwell GPU detected (sm_' + cc + ') - using native cuTile')
    exit(1)
else:
    print('[cuTile Compat] Non-Blackwell GPU detected (sm_' + cc + ') - using compatibility layer')
    exit(0)
" 2>/dev/null
    return $?
}

# If script is sourced, just set up environment
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "[hack.sh] Environment configured for cuTile & hw1-asr"
    echo "          PYTHONPATH includes: ${HACK_DIR}"
    echo "          CUDA_HOME: ${CUDA_HOME}"
    echo "          PROJECT_ROOT: ${PROJECT_ROOT}"
    check_gpu
    return 0
fi

# If script is executed with arguments, run the Python script
if [[ $# -gt 0 ]]; then
    # Check GPU first
    check_gpu
    USE_COMPAT=$?

    if [[ $USE_COMPAT -eq 1 ]]; then
        # Blackwell GPU - use native cuTile without hack
        echo "[hack.sh] Using native cuTile (Blackwell GPU detected)"
        PYTHONPATH="" python3 "$@"
    else
        # Non-Blackwell GPU - use compatibility layer
        echo "[hack.sh] Using compatibility layer"
        echo ""
        python3 "$@"
    fi
else
    echo "cuTile Hopper Hack - Run cuTile & hw1-asr on non-Blackwell GPUs"
    echo ""
    echo "Usage:"
    echo "  $0 <script.py>        Run a Python script with the compatibility layer"
    echo "  source $0             Set up environment for current shell (RECOMMENDED)"
    echo ""
    echo "Examples (after sourcing):"
    echo "  # cuTile tutorials"
    echo "  python cutile-tutorial/1-vectoradd/vectoradd.py"
    echo "  python cutile-tutorial/7-attention/attention.py"
    echo ""
    echo "  # hw1-asr benchmarks"
    echo "  python hw1-asr/benchmark_student.py glm_asr_scratch"
    echo "  python hw1-asr/benchmark_student.py glm_asr_cutile_example"
    echo ""
    echo "Supported:"
    echo "  cuTile tutorials: 1-vectoradd, 2-execution-model, 3-data-model,"
    echo "                    4-transpose, 6-performance-tuning, 7-attention"
    echo "  hw1-asr:          glm_asr_scratch, glm_asr_cutile_*"
fi
