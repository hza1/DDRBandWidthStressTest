#!/bin/bash
#
# run_test.sh - DDR Bandwidth Stress Test Runner for Nvidia Thor SoC
#
# Usage:
#   ./run_test.sh [all|cpu|gpu|mixed]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/ddr_bw_stress"
LOG_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/ddr_bw_result_${TIMESTAMP}.log"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if binary exists
if [ ! -f "${BINARY}" ]; then
    print_warn "Binary not found. Building..."
    cd "${SCRIPT_DIR}"
    make
    if [ $? -ne 0 ]; then
        print_error "Build failed!"
        exit 1
    fi
    print_info "Build successful."
fi

# Create results directory
mkdir -p "${LOG_DIR}"

# System info
print_info "Collecting system information..."
echo "========================================" | tee "${LOG_FILE}"
echo " System Information" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "Date:     $(date)" | tee -a "${LOG_FILE}"
echo "Hostname: $(hostname)" | tee -a "${LOG_FILE}"
echo "Kernel:   $(uname -r)" | tee -a "${LOG_FILE}"
echo "Arch:     $(uname -m)" | tee -a "${LOG_FILE}"

if [ -f /proc/cpuinfo ]; then
    CPU_MODEL=$(grep -m1 "model name\|CPU part" /proc/cpuinfo | head -1)
    echo "CPU:      ${CPU_MODEL}" | tee -a "${LOG_FILE}"
fi

CPU_CORES=$(nproc)
echo "CPU Cores: ${CPU_CORES}" | tee -a "${LOG_FILE}"

# Memory info
MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
echo "Memory:   ${MEM_TOTAL}" | tee -a "${LOG_FILE}"

# GPU info
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "GPU:      ${GPU_NAME}" | tee -a "${LOG_FILE}"
fi

echo "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Determine test mode
TEST_MODE="${1:-all}"

case "${TEST_MODE}" in
    all)
        print_info "Running ALL tests (CPU + GPU + Mixed)..."
        print_info "Estimated total time: ~9 minutes (3 sub-tests x 60s x 3 modes)"
        "${BINARY}" 2>&1 | tee -a "${LOG_FILE}"
        ;;
    cpu)
        print_info "Running CPU-only test..."
        print_info "Estimated time: ~3 minutes (3 sub-tests x 60s)"
        "${BINARY}" --cpu 2>&1 | tee -a "${LOG_FILE}"
        ;;
    gpu)
        print_info "Running GPU-only test..."
        print_info "Estimated time: ~3 minutes (3 sub-tests x 60s)"
        "${BINARY}" --gpu 2>&1 | tee -a "${LOG_FILE}"
        ;;
    mixed)
        print_info "Running CPU+GPU mixed test..."
        print_info "Estimated time: ~3 minutes (3 sub-tests x 60s)"
        "${BINARY}" --mixed 2>&1 | tee -a "${LOG_FILE}"
        ;;
    *)
        print_error "Unknown test mode: ${TEST_MODE}"
        echo "Usage: $0 [all|cpu|gpu|mixed]"
        exit 1
        ;;
esac

echo "" | tee -a "${LOG_FILE}"
print_info "Results saved to: ${LOG_FILE}"
