#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "filters.h"

// ── Error helpers ─────────────────────────────────────────────────────────────
#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t _e = (call);                                             \
        if (_e != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s at %s:%d\n",                     \
                    hipGetErrorString(_e), __FILE__, __LINE__);             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static float event_ms(hipEvent_t a, hipEvent_t b) {
    float ms = 0;
    hipEventElapsedTime(&ms, a, b);
    return ms;
}

// ── Constants & limits ────────────────────────────────────────────────────────
static constexpr int BLOCK_DIM  = 16;
static constexpr int MAX_RADIUS = 8;           // supports up to 17×17 kernel

// Gaussian kernel lives in constant memory — broadcast-read on every thread.
__constant__ float dc_kernel[(2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1)];

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: grayscale
// Each thread processes one pixel; luminosity formula (BT.601).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_grayscale(const uint8_t* __restrict__ src,
                             uint8_t* __restrict__       dst,
                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int i    = (y * width + x) * 3;
    auto g   = static_cast<uint8_t>(0.299f * src[i] +
                                     0.587f * src[i + 1] +
                                     0.114f * src[i + 2]);
    dst[i] = dst[i + 1] = dst[i + 2] = g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: Gaussian blur (shared-memory tiled)
//
// Each block loads a (BLOCK_DIM + 2*r) × (BLOCK_DIM + 2*r) RGB tile into
// shared memory (with clamped-border halo), then every active thread
// accumulates the convolution sum using the pre-loaded dc_kernel weights.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_gaussian(const uint8_t* __restrict__ src,
                            uint8_t* __restrict__       dst,
                            int width, int height, int radius)
{
    extern __shared__ uint8_t smem[];  // (BLOCK_DIM+2r)*(BLOCK_DIM+2r)*3 bytes

    const int tile_w = blockDim.x + 2 * radius;
    const int tile_h = blockDim.y + 2 * radius;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // ── Cooperative tile load (each thread covers multiple smem cells) ───────
    for (int dy = ty; dy < tile_h; dy += blockDim.y) {
        for (int dx = tx; dx < tile_w; dx += blockDim.x) {
            int img_x = min(max(static_cast<int>(blockIdx.x * blockDim.x) + dx - radius, 0), width  - 1);
            int img_y = min(max(static_cast<int>(blockIdx.y * blockDim.y) + dy - radius, 0), height - 1);
            int si = (dy * tile_w + dx) * 3;
            int gi = (img_y * width + img_x) * 3;
            smem[si]     = src[gi];
            smem[si + 1] = src[gi + 1];
            smem[si + 2] = src[gi + 2];
        }
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    if (x >= width || y >= height) return;

    float r = 0, g = 0, b = 0;
    int ksize = 2 * radius + 1;

    for (int ky = 0; ky < ksize; ky++) {
        for (int kx = 0; kx < ksize; kx++) {
            float w  = dc_kernel[ky * ksize + kx];
            int   si = ((ty + ky) * tile_w + (tx + kx)) * 3;
            r += smem[si]     * w;
            g += smem[si + 1] * w;
            b += smem[si + 2] * w;
        }
    }

    int oi = (y * width + x) * 3;
    dst[oi]     = static_cast<uint8_t>(fminf(fmaxf(r, 0.f), 255.f));
    dst[oi + 1] = static_cast<uint8_t>(fminf(fmaxf(g, 0.f), 255.f));
    dst[oi + 2] = static_cast<uint8_t>(fminf(fmaxf(b, 0.f), 255.f));
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: Sobel edge detection (shared-memory tiled, radius=1)
//
// Converts each pixel to grayscale in shared memory, then applies the 3×3
// Sobel operators Gx and Gy; the output magnitude is written to all channels.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_sobel(const uint8_t* __restrict__ src,
                         uint8_t* __restrict__       dst,
                         int width, int height)
{
    extern __shared__ uint8_t smem[];  // (BLOCK_DIM+2)*(BLOCK_DIM+2) grayscale bytes

    const int tile_w = blockDim.x + 2;
    const int tile_h = blockDim.y + 2;
    const int tx = threadIdx.x, ty = threadIdx.y;

    for (int dy = ty; dy < tile_h; dy += blockDim.y) {
        for (int dx = tx; dx < tile_w; dx += blockDim.x) {
            int img_x = min(max(static_cast<int>(blockIdx.x * blockDim.x) + dx - 1, 0), width  - 1);
            int img_y = min(max(static_cast<int>(blockIdx.y * blockDim.y) + dy - 1, 0), height - 1);
            int gi = (img_y * width + img_x) * 3;
            smem[dy * tile_w + dx] = static_cast<uint8_t>(
                0.299f * src[gi] + 0.587f * src[gi + 1] + 0.114f * src[gi + 2]);
        }
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    if (x >= width || y >= height) return;

    // 3×3 neighbourhood (row-major, origin at top-left of halo window)
    float p00 = smem[(ty    ) * tile_w + (tx    )];
    float p01 = smem[(ty    ) * tile_w + (tx + 1)];
    float p02 = smem[(ty    ) * tile_w + (tx + 2)];
    float p10 = smem[(ty + 1) * tile_w + (tx    )];
    float p12 = smem[(ty + 1) * tile_w + (tx + 2)];
    float p20 = smem[(ty + 2) * tile_w + (tx    )];
    float p21 = smem[(ty + 2) * tile_w + (tx + 1)];
    float p22 = smem[(ty + 2) * tile_w + (tx + 2)];

    float gx  = -p00 + p02 - 2.f * p10 + 2.f * p12 - p20 + p22;
    float gy  = -p00 - 2.f * p01 - p02 + p20 + 2.f * p21 + p22;
    auto  mag = static_cast<uint8_t>(fminf(sqrtf(gx * gx + gy * gy), 255.f));

    int oi = (y * width + x) * 3;
    dst[oi] = dst[oi + 1] = dst[oi + 2] = mag;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: brightness / contrast  — dst = src * contrast + brightness
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_brightness_contrast(const uint8_t* __restrict__ src,
                                       uint8_t* __restrict__       dst,
                                       int width, int height,
                                       float brightness, float contrast)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int i = (y * width + x) * 3;
    for (int c = 0; c < 3; c++) {
        float v  = src[i + c] * contrast + brightness;
        dst[i + c] = static_cast<uint8_t>(fminf(fmaxf(v, 0.f), 255.f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host utilities
// ─────────────────────────────────────────────────────────────────────────────
static void build_gaussian_kernel(float* out, int radius, float sigma) {
    int    sz  = 2 * radius + 1;
    float  sum = 0.f;
    for (int y = -radius; y <= radius; y++)
        for (int x = -radius; x <= radius; x++) {
            float v = expf(-(float)(x * x + y * y) / (2.f * sigma * sigma));
            out[(y + radius) * sz + (x + radius)] = v;
            sum += v;
        }
    for (int i = 0; i < sz * sz; i++) out[i] /= sum;
}

// Allocate device buffers, run kernel via HIP events, return timing.
struct GpuCtx {
    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    hipEvent_t ev[4];   // 0=start_h2d, 1=end_h2d/start_k, 2=end_k/start_d2h, 3=end_d2h

    GpuCtx(size_t bytes) {
        HIP_CHECK(hipMalloc(&d_src, bytes));
        HIP_CHECK(hipMalloc(&d_dst, bytes));
        for (auto& e : ev) HIP_CHECK(hipEventCreate(&e));
    }
    ~GpuCtx() {
        hipFree(d_src); hipFree(d_dst);
        for (auto& e : ev) hipEventDestroy(e);
    }
};

static void fill_timing(TimingMs* t, const GpuCtx& ctx) {
    if (!t) return;
    t->h2d    = event_ms(ctx.ev[0], ctx.ev[1]);
    t->kernel = event_ms(ctx.ev[1], ctx.ev[2]);
    t->d2h    = event_ms(ctx.ev[2], ctx.ev[3]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public GPU wrappers
// ─────────────────────────────────────────────────────────────────────────────
void gpu_grayscale(const uint8_t* h_src, uint8_t* h_dst,
                   int width, int height, TimingMs* t)
{
    size_t bytes = (size_t)width * height * 3;
    GpuCtx ctx(bytes);

    HIP_CHECK(hipEventRecord(ctx.ev[0]));
    HIP_CHECK(hipMemcpy(ctx.d_src, h_src, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(ctx.ev[1]));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    hipLaunchKernelGGL(k_grayscale, grid, block, 0, 0, ctx.d_src, ctx.d_dst, width, height);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(ctx.ev[2]));

    HIP_CHECK(hipMemcpy(h_dst, ctx.d_dst, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventRecord(ctx.ev[3]));
    HIP_CHECK(hipEventSynchronize(ctx.ev[3]));

    fill_timing(t, ctx);
}

void gpu_gaussian_blur(const uint8_t* h_src, uint8_t* h_dst,
                       int width, int height, float sigma, int radius, TimingMs* t)
{
    if (radius < 1) radius = 1;
    if (radius > MAX_RADIUS) radius = MAX_RADIUS;

    float h_kernel[17 * 17] = {};
    build_gaussian_kernel(h_kernel, radius, sigma);
    int ksize = 2 * radius + 1;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(dc_kernel), h_kernel,
                                sizeof(float) * ksize * ksize, 0,
                                hipMemcpyHostToDevice));

    size_t bytes = (size_t)width * height * 3;
    GpuCtx ctx(bytes);

    HIP_CHECK(hipEventRecord(ctx.ev[0]));
    HIP_CHECK(hipMemcpy(ctx.d_src, h_src, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(ctx.ev[1]));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    size_t smem = (size_t)(BLOCK_DIM + 2 * radius) * (BLOCK_DIM + 2 * radius) * 3;
    hipLaunchKernelGGL(k_gaussian, grid, block, smem, 0,
                       ctx.d_src, ctx.d_dst, width, height, radius);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(ctx.ev[2]));

    HIP_CHECK(hipMemcpy(h_dst, ctx.d_dst, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventRecord(ctx.ev[3]));
    HIP_CHECK(hipEventSynchronize(ctx.ev[3]));

    fill_timing(t, ctx);
}

void gpu_sobel(const uint8_t* h_src, uint8_t* h_dst,
               int width, int height, TimingMs* t)
{
    size_t bytes = (size_t)width * height * 3;
    GpuCtx ctx(bytes);

    HIP_CHECK(hipEventRecord(ctx.ev[0]));
    HIP_CHECK(hipMemcpy(ctx.d_src, h_src, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(ctx.ev[1]));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    size_t smem = (size_t)(BLOCK_DIM + 2) * (BLOCK_DIM + 2);
    hipLaunchKernelGGL(k_sobel, grid, block, smem, 0, ctx.d_src, ctx.d_dst, width, height);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(ctx.ev[2]));

    HIP_CHECK(hipMemcpy(h_dst, ctx.d_dst, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventRecord(ctx.ev[3]));
    HIP_CHECK(hipEventSynchronize(ctx.ev[3]));

    fill_timing(t, ctx);
}

void gpu_brightness_contrast(const uint8_t* h_src, uint8_t* h_dst,
                              int width, int height,
                              float brightness, float contrast, TimingMs* t)
{
    size_t bytes = (size_t)width * height * 3;
    GpuCtx ctx(bytes);

    HIP_CHECK(hipEventRecord(ctx.ev[0]));
    HIP_CHECK(hipMemcpy(ctx.d_src, h_src, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(ctx.ev[1]));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    hipLaunchKernelGGL(k_brightness_contrast, grid, block, 0, 0,
                       ctx.d_src, ctx.d_dst, width, height, brightness, contrast);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(ctx.ev[2]));

    HIP_CHECK(hipMemcpy(h_dst, ctx.d_dst, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventRecord(ctx.ev[3]));
    HIP_CHECK(hipEventSynchronize(ctx.ev[3]));

    fill_timing(t, ctx);
}
