#pragma once
#include <cstdint>

// Timing breakdown returned by GPU wrappers (all in milliseconds).
struct TimingMs {
    float h2d    = 0;  // host→device transfer
    float kernel = 0;  // kernel execution
    float d2h    = 0;  // device→host transfer
};

// ── GPU filters (HIP/ROCm) ────────────────────────────────────────────────────
// All images: interleaved RGB uint8, row-major (width*3 bytes per row).

void gpu_grayscale(const uint8_t* src, uint8_t* dst,
                   int width, int height, TimingMs* t = nullptr);

void gpu_gaussian_blur(const uint8_t* src, uint8_t* dst,
                       int width, int height,
                       float sigma, int radius,
                       TimingMs* t = nullptr);

void gpu_sobel(const uint8_t* src, uint8_t* dst,
               int width, int height, TimingMs* t = nullptr);

void gpu_brightness_contrast(const uint8_t* src, uint8_t* dst,
                              int width, int height,
                              float brightness, float contrast,
                              TimingMs* t = nullptr);

// ── CPU reference implementations (single-threaded) ──────────────────────────
void cpu_grayscale(const uint8_t* src, uint8_t* dst, int width, int height);
void cpu_gaussian_blur(const uint8_t* src, uint8_t* dst, int width, int height, float sigma, int radius);
void cpu_sobel(const uint8_t* src, uint8_t* dst, int width, int height);
void cpu_brightness_contrast(const uint8_t* src, uint8_t* dst, int width, int height, float brightness, float contrast);
