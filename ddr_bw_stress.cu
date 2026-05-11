/*
 * DDR Bandwidth Stress Test for Nvidia Thor SoC
 * Target: ARM64, Ubuntu 20.04, CUDA
 *
 * Tests:
 *   1. CPU-only DDR bandwidth (multi-threaded NEON streaming)
 *   2. GPU-only DDR bandwidth (CUDA kernel streaming)
 *   3. CPU+GPU mixed DDR bandwidth (simultaneous)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <arm_neon.h>
#include <cuda_runtime.h>

#define DATA_BLOCK_SIZE     (2ULL * 1024 * 1024 * 1024)  /* 2GB */
#define TEST_DURATION_SEC   60
#define CPU_NUM_THREADS     8

/* ============================================================
 * Utility
 * ============================================================ */

static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* ============================================================
 * CPU Bandwidth Test (ARM64 NEON)
 * ============================================================ */

typedef struct {
    volatile int *running;
    uint8_t *buf;
    size_t size;
    double bytes_transferred;
} cpu_thread_arg_t;

static void *cpu_read_thread(void *arg)
{
    cpu_thread_arg_t *ta = (cpu_thread_arg_t *)arg;
    uint8_t *buf = ta->buf;
    size_t size = ta->size;
    double total = 0.0;

    while (*(ta->running)) {
        uint8x16_t sum = vdupq_n_u8(0);
        for (size_t i = 0; i < size; i += 64) {
            uint8x16_t v0 = vld1q_u8(buf + i);
            uint8x16_t v1 = vld1q_u8(buf + i + 16);
            uint8x16_t v2 = vld1q_u8(buf + i + 32);
            uint8x16_t v3 = vld1q_u8(buf + i + 48);
            sum = vaddq_u8(sum, v0);
            sum = vaddq_u8(sum, v1);
            sum = vaddq_u8(sum, v2);
            sum = vaddq_u8(sum, v3);
        }
        /* Prevent optimization */
        volatile uint8_t sink;
        sink = vgetq_lane_u8(sum, 0);
        (void)sink;
        total += (double)size;
    }

    ta->bytes_transferred = total;
    return NULL;
}

static void *cpu_write_thread(void *arg)
{
    cpu_thread_arg_t *ta = (cpu_thread_arg_t *)arg;
    uint8_t *buf = ta->buf;
    size_t size = ta->size;
    double total = 0.0;
    uint8x16_t pattern = vdupq_n_u8(0xAA);

    while (*(ta->running)) {
        for (size_t i = 0; i < size; i += 64) {
            vst1q_u8(buf + i, pattern);
            vst1q_u8(buf + i + 16, pattern);
            vst1q_u8(buf + i + 32, pattern);
            vst1q_u8(buf + i + 48, pattern);
        }
        total += (double)size;
    }

    ta->bytes_transferred = total;
    return NULL;
}

static void *cpu_copy_thread(void *arg)
{
    cpu_thread_arg_t *ta = (cpu_thread_arg_t *)arg;
    uint8_t *buf = ta->buf;
    size_t size = ta->size;
    size_t half = size / 2;
    uint8_t *src = buf;
    uint8_t *dst = buf + half;
    double total = 0.0;

    while (*(ta->running)) {
        for (size_t i = 0; i < half; i += 64) {
            uint8x16_t v0 = vld1q_u8(src + i);
            uint8x16_t v1 = vld1q_u8(src + i + 16);
            uint8x16_t v2 = vld1q_u8(src + i + 32);
            uint8x16_t v3 = vld1q_u8(src + i + 48);
            vst1q_u8(dst + i, v0);
            vst1q_u8(dst + i + 16, v1);
            vst1q_u8(dst + i + 32, v2);
            vst1q_u8(dst + i + 48, v3);
        }
        /* copy counts as read + write */
        total += (double)size;
    }

    ta->bytes_transferred = total;
    return NULL;
}

typedef struct {
    double read_bw;
    double write_bw;
    double copy_bw;
} cpu_bw_result_t;

static cpu_bw_result_t run_cpu_bandwidth_test(void)
{
    cpu_bw_result_t result = {0};
    int num_threads = CPU_NUM_THREADS;
    size_t per_thread_size = DATA_BLOCK_SIZE / num_threads;

    /* Align to 64 bytes */
    per_thread_size &= ~63ULL;

    printf("  [CPU] Allocating %llu MB total (%d threads x %llu MB)\n",
           (unsigned long long)(per_thread_size * num_threads) / (1024*1024),
           num_threads,
           (unsigned long long)per_thread_size / (1024*1024));

    uint8_t **bufs = (uint8_t **)malloc(num_threads * sizeof(uint8_t *));
    for (int i = 0; i < num_threads; i++) {
        posix_memalign((void **)&bufs[i], 64, per_thread_size);
        if (!bufs[i]) {
            fprintf(stderr, "Failed to allocate CPU buffer\n");
            exit(EXIT_FAILURE);
        }
        memset(bufs[i], 0x55, per_thread_size);
    }

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    cpu_thread_arg_t *args = (cpu_thread_arg_t *)malloc(num_threads * sizeof(cpu_thread_arg_t));
    volatile int running;

    /* --- Read test --- */
    printf("  [CPU] Running READ test for %d seconds...\n", TEST_DURATION_SEC);
    running = 1;
    for (int i = 0; i < num_threads; i++) {
        args[i].running = &running;
        args[i].buf = bufs[i];
        args[i].size = per_thread_size;
        args[i].bytes_transferred = 0;
        pthread_create(&threads[i], NULL, cpu_read_thread, &args[i]);
    }

    double t0 = get_time_sec();
    sleep(TEST_DURATION_SEC);
    running = 0;
    double elapsed = get_time_sec() - t0;

    double total_bytes = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_bytes += args[i].bytes_transferred;
    }
    result.read_bw = total_bytes / elapsed / (1024.0 * 1024.0 * 1024.0);
    printf("  [CPU] READ bandwidth:  %.2f GB/s\n", result.read_bw);

    /* --- Write test --- */
    printf("  [CPU] Running WRITE test for %d seconds...\n", TEST_DURATION_SEC);
    running = 1;
    for (int i = 0; i < num_threads; i++) {
        args[i].running = &running;
        args[i].buf = bufs[i];
        args[i].size = per_thread_size;
        args[i].bytes_transferred = 0;
        pthread_create(&threads[i], NULL, cpu_write_thread, &args[i]);
    }

    t0 = get_time_sec();
    sleep(TEST_DURATION_SEC);
    running = 0;
    elapsed = get_time_sec() - t0;

    total_bytes = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_bytes += args[i].bytes_transferred;
    }
    result.write_bw = total_bytes / elapsed / (1024.0 * 1024.0 * 1024.0);
    printf("  [CPU] WRITE bandwidth: %.2f GB/s\n", result.write_bw);

    /* --- Copy test --- */
    printf("  [CPU] Running COPY test for %d seconds...\n", TEST_DURATION_SEC);
    running = 1;
    for (int i = 0; i < num_threads; i++) {
        args[i].running = &running;
        args[i].buf = bufs[i];
        args[i].size = per_thread_size;
        args[i].bytes_transferred = 0;
        pthread_create(&threads[i], NULL, cpu_copy_thread, &args[i]);
    }

    t0 = get_time_sec();
    sleep(TEST_DURATION_SEC);
    running = 0;
    elapsed = get_time_sec() - t0;

    total_bytes = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_bytes += args[i].bytes_transferred;
    }
    result.copy_bw = total_bytes / elapsed / (1024.0 * 1024.0 * 1024.0);
    printf("  [CPU] COPY bandwidth:  %.2f GB/s\n", result.copy_bw);

    for (int i = 0; i < num_threads; i++)
        free(bufs[i]);
    free(bufs);
    free(threads);
    free(args);

    return result;
}

/* ============================================================
 * GPU Bandwidth Test (CUDA)
 *
 * Strategy: use driver-level APIs (cudaMemcpy/cudaMemset) which
 * invoke the hardware DMA/copy engine and CANNOT be optimized.
 * Also includes a kernel-based read test with full verification.
 * ============================================================ */

/* Read kernel: reads src, transforms each element, writes to dst.
 * Every element in dst gets a unique value derived from src, so
 * the compiler cannot eliminate any reads or writes. */
__global__ void gpu_read_write_kernel(const int4 *__restrict__ src,
                                      int4 *__restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        int4 v = src[i];
        /* XOR transform ensures compiler cannot predict or eliminate */
        v.x ^= 0xDEADBEEF;
        v.y ^= 0xCAFEBABE;
        v.z ^= 0x12345678;
        v.w ^= 0x9ABCDEF0;
        dst[i] = v;
    }
}

typedef struct {
    double read_bw;
    double write_bw;
    double copy_bw;
} gpu_bw_result_t;

static gpu_bw_result_t run_gpu_bandwidth_test(void)
{
    gpu_bw_result_t result = {0};
    size_t alloc_size = DATA_BLOCK_SIZE;
    size_t n_int4 = alloc_size / sizeof(int4);

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("  [GPU] Device: %s\n", prop.name);
    printf("  [GPU] Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  [GPU] Allocating 2 x %llu MB on device\n",
           (unsigned long long)alloc_size / (1024*1024));

    int4 *d_buf1, *d_buf2;
    CUDA_CHECK(cudaMalloc(&d_buf1, alloc_size));
    CUDA_CHECK(cudaMalloc(&d_buf2, alloc_size));
    CUDA_CHECK(cudaMemset(d_buf1, 0xAA, alloc_size));
    CUDA_CHECK(cudaMemset(d_buf2, 0x00, alloc_size));

    int block_size = 256;
    int num_blocks = prop.multiProcessorCount * 32;

    printf("  [GPU] L2 cache size: %d KB, SMs: %d, grid: %d blocks x %d threads\n",
           prop.l2CacheSize / 1024, prop.multiProcessorCount, num_blocks, block_size);

    /* --- Verify kernel actually works --- */
    printf("  [GPU] Verifying kernel execution...\n");
    gpu_read_write_kernel<<<num_blocks, block_size>>>(d_buf1, d_buf2, n_int4);
    CUDA_CHECK(cudaGetLastError());  /* catch launch errors (e.g. wrong arch) */
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        int4 h_check;
        CUDA_CHECK(cudaMemcpy(&h_check, d_buf2, sizeof(int4), cudaMemcpyDeviceToHost));
        int4 h_src;
        CUDA_CHECK(cudaMemcpy(&h_src, d_buf1, sizeof(int4), cudaMemcpyDeviceToHost));
        int4 expected;
        expected.x = h_src.x ^ 0xDEADBEEF;
        expected.y = h_src.y ^ 0xCAFEBABE;
        expected.z = h_src.z ^ 0x12345678;
        expected.w = h_src.w ^ 0x9ABCDEF0;
        if (h_check.x != expected.x || h_check.y != expected.y ||
            h_check.z != expected.z || h_check.w != expected.w) {
            printf("  [GPU] WARNING: Kernel verification FAILED! Results may be invalid.\n");
            printf("  [GPU]   Expected: 0x%08X 0x%08X 0x%08X 0x%08X\n",
                   expected.x, expected.y, expected.z, expected.w);
            printf("  [GPU]   Got:      0x%08X 0x%08X 0x%08X 0x%08X\n",
                   h_check.x, h_check.y, h_check.z, h_check.w);
        } else {
            printf("  [GPU] Kernel verification PASSED.\n");
        }
    }

    unsigned int iteration;
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* --- Read test (kernel-based: read src + write dst) --- */
    printf("  [GPU] Running READ test for %d seconds (kernel: read+transform+write)...\n",
           TEST_DURATION_SEC);
    {
        double total_bytes = 0;
        iteration = 0;
        CUDA_CHECK(cudaMemset(d_buf2, 0x00, alloc_size));
        CUDA_CHECK(cudaDeviceSynchronize());

        double t0 = get_time_sec();
        while (get_time_sec() - t0 < TEST_DURATION_SEC) {
            gpu_read_write_kernel<<<num_blocks, block_size>>>(d_buf1, d_buf2, n_int4);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            total_bytes += (double)alloc_size;  /* count only read bytes */
            iteration++;
        }
        double elapsed = get_time_sec() - t0;
        result.read_bw = total_bytes / elapsed / (1024.0 * 1024.0 * 1024.0);
        printf("  [GPU] READ bandwidth:  %.2f GB/s (%u iterations, %.2f ms/iter)\n",
               result.read_bw, iteration, elapsed * 1000.0 / iteration);
    }

    /* --- Write test (cudaMemset - hardware DMA engine) --- */
    printf("  [GPU] Running WRITE test for %d seconds (cudaMemset)...\n",
           TEST_DURATION_SEC);
    {
        double total_bytes = 0;
        iteration = 0;
        double t0 = get_time_sec();
        while (get_time_sec() - t0 < TEST_DURATION_SEC) {
            CUDA_CHECK(cudaMemset(d_buf1, (int)(iteration & 0xFF), alloc_size));
            CUDA_CHECK(cudaDeviceSynchronize());
            total_bytes += (double)alloc_size;
            iteration++;
        }
        double elapsed = get_time_sec() - t0;
        result.write_bw = total_bytes / elapsed / (1024.0 * 1024.0 * 1024.0);
        printf("  [GPU] WRITE bandwidth: %.2f GB/s (%u iterations, %.2f ms/iter)\n",
               result.write_bw, iteration, elapsed * 1000.0 / iteration);
    }

    /* --- Copy test (cudaMemcpy D2D - hardware copy engine) --- */
    printf("  [GPU] Running COPY test for %d seconds (cudaMemcpy D2D)...\n",
           TEST_DURATION_SEC);
    {
        double total_bytes = 0;
        iteration = 0;
        double t0 = get_time_sec();
        while (get_time_sec() - t0 < TEST_DURATION_SEC) {
            CUDA_CHECK(cudaMemcpy(d_buf2, d_buf1, alloc_size, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            total_bytes += (double)alloc_size * 2;  /* read + write */
            iteration++;
        }
        double elapsed = get_time_sec() - t0;
        result.copy_bw = total_bytes / elapsed / (1024.0 * 1024.0 * 1024.0);
        printf("  [GPU] COPY bandwidth:  %.2f GB/s (%u iterations, %.2f ms/iter)\n",
               result.copy_bw, iteration, elapsed * 1000.0 / iteration);
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaFree(d_buf2));

    return result;
}

/* ============================================================
 * Mixed CPU+GPU Bandwidth Test
 * ============================================================ */

typedef struct {
    gpu_bw_result_t gpu_result;
} gpu_thread_result_t;

static void *mixed_gpu_thread(void *arg)
{
    gpu_thread_result_t *res = (gpu_thread_result_t *)arg;
    res->gpu_result = run_gpu_bandwidth_test();
    return NULL;
}

typedef struct {
    cpu_bw_result_t cpu_result;
} cpu_thread_result_t;

static void *mixed_cpu_thread(void *arg)
{
    cpu_thread_result_t *res = (cpu_thread_result_t *)arg;
    res->cpu_result = run_cpu_bandwidth_test();
    return NULL;
}

/* ============================================================
 * Main
 * ============================================================ */

static void print_separator(void)
{
    printf("================================================================\n");
}

static void print_results_summary(const char *test_name,
                                  double read_bw, double write_bw, double copy_bw)
{
    printf("  %-10s READ:  %8.2f GB/s\n", test_name, read_bw);
    printf("  %-10s WRITE: %8.2f GB/s\n", test_name, write_bw);
    printf("  %-10s COPY:  %8.2f GB/s\n", test_name, copy_bw);
}

int main(int argc, char *argv[])
{
    int run_cpu = 0, run_gpu = 0, run_mixed = 0;

    if (argc < 2) {
        run_cpu = run_gpu = run_mixed = 1;
    } else {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--cpu") == 0) run_cpu = 1;
            else if (strcmp(argv[i], "--gpu") == 0) run_gpu = 1;
            else if (strcmp(argv[i], "--mixed") == 0) run_mixed = 1;
            else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [--cpu] [--gpu] [--mixed] [--help]\n", argv[0]);
                printf("  --cpu    Run CPU-only DDR bandwidth test\n");
                printf("  --gpu    Run GPU-only DDR bandwidth test\n");
                printf("  --mixed  Run CPU+GPU mixed DDR bandwidth test\n");
                printf("  (no args = run all tests)\n");
                return 0;
            }
        }
    }

    printf("\n");
    print_separator();
    printf("  DDR Bandwidth Stress Test - Nvidia Thor SoC\n");
    printf("  Data block size: %llu MB\n",
           (unsigned long long)DATA_BLOCK_SIZE / (1024*1024));
    printf("  Test duration:   %d seconds per sub-test\n", TEST_DURATION_SEC);
    printf("  CPU threads:     %d\n", CPU_NUM_THREADS);
    print_separator();
    printf("\n");

    cpu_bw_result_t cpu_result = {0};
    gpu_bw_result_t gpu_result = {0};
    cpu_bw_result_t mixed_cpu_result = {0};
    gpu_bw_result_t mixed_gpu_result = {0};

    /* Test 1: CPU only */
    if (run_cpu) {
        printf("[TEST 1] CPU-Only DDR Bandwidth\n");
        print_separator();
        cpu_result = run_cpu_bandwidth_test();
        printf("\n");
    }

    /* Test 2: GPU only */
    if (run_gpu) {
        printf("[TEST 2] GPU-Only DDR Bandwidth\n");
        print_separator();
        gpu_result = run_gpu_bandwidth_test();
        printf("\n");
    }

    /* Test 3: CPU + GPU mixed */
    if (run_mixed) {
        printf("[TEST 3] CPU+GPU Mixed DDR Bandwidth\n");
        print_separator();
        printf("  Running CPU and GPU tests simultaneously...\n\n");

        pthread_t cpu_tid, gpu_tid;
        cpu_thread_result_t cpu_res;
        gpu_thread_result_t gpu_res;

        pthread_create(&gpu_tid, NULL, mixed_gpu_thread, &gpu_res);
        pthread_create(&cpu_tid, NULL, mixed_cpu_thread, &cpu_res);

        pthread_join(cpu_tid, NULL);
        pthread_join(gpu_tid, NULL);

        mixed_cpu_result = cpu_res.cpu_result;
        mixed_gpu_result = gpu_res.gpu_result;
        printf("\n");
    }

    /* Summary */
    print_separator();
    printf("  RESULTS SUMMARY\n");
    print_separator();
    printf("\n");

    if (run_cpu) {
        printf("  [CPU-Only Test]\n");
        print_results_summary("[CPU]", cpu_result.read_bw,
                              cpu_result.write_bw, cpu_result.copy_bw);
        printf("\n");
    }

    if (run_gpu) {
        printf("  [GPU-Only Test]\n");
        print_results_summary("[GPU]", gpu_result.read_bw,
                              gpu_result.write_bw, gpu_result.copy_bw);
        printf("\n");
    }

    if (run_mixed) {
        printf("  [CPU+GPU Mixed Test]\n");
        print_results_summary("[CPU]", mixed_cpu_result.read_bw,
                              mixed_cpu_result.write_bw, mixed_cpu_result.copy_bw);
        print_results_summary("[GPU]", mixed_gpu_result.read_bw,
                              mixed_gpu_result.write_bw, mixed_gpu_result.copy_bw);
        double total_read = mixed_cpu_result.read_bw + mixed_gpu_result.read_bw;
        double total_write = mixed_cpu_result.write_bw + mixed_gpu_result.write_bw;
        double total_copy = mixed_cpu_result.copy_bw + mixed_gpu_result.copy_bw;
        printf("  %-10s READ:  %8.2f GB/s (combined)\n", "[TOTAL]", total_read);
        printf("  %-10s WRITE: %8.2f GB/s (combined)\n", "[TOTAL]", total_write);
        printf("  %-10s COPY:  %8.2f GB/s (combined)\n", "[TOTAL]", total_copy);
        printf("\n");
    }

    print_separator();
    printf("  Test completed.\n");
    print_separator();
    printf("\n");

    return 0;
}
