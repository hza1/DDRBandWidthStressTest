# DDR Bandwidth Stress Test for Nvidia Thor SoC
# Target: ARM64 (aarch64), Ubuntu 20.04, CUDA

NVCC        ?= /usr/local/cuda/bin/nvcc
CUDA_PATH   ?= /usr/local/cuda

# Thor SoC GPU architecture (Ampere-based, SM 8.7)
GPU_ARCH    ?= sm_87

CFLAGS      = -O3 -march=armv8-a+simd
NVCCFLAGS   = -O3 -arch=$(GPU_ARCH) -Xcompiler "$(CFLAGS)" --std=c++11
LDFLAGS     = -lpthread -lrt

TARGET      = ddr_bw_stress
SRC         = ddr_bw_stress.cu

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

help:
	@echo "DDR Bandwidth Stress Test - Nvidia Thor SoC"
	@echo ""
	@echo "Build:"
	@echo "  make              - Build with default settings"
	@echo "  make GPU_ARCH=sm_87  - Specify GPU architecture"
	@echo "  make CUDA_PATH=/path/to/cuda - Specify CUDA path"
	@echo ""
	@echo "Run:"
	@echo "  ./ddr_bw_stress          - Run all tests"
	@echo "  ./ddr_bw_stress --cpu    - CPU-only test"
	@echo "  ./ddr_bw_stress --gpu    - GPU-only test"
	@echo "  ./ddr_bw_stress --mixed  - CPU+GPU mixed test"
