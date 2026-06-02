#include "filters.h"
#include <cmath>
#include <cstring>
#include <algorithm>

using std::min;
using std::max;

static uint8_t clamp_u8(float v) {
    return static_cast<uint8_t>(std::min(std::max(v, 0.f), 255.f));
}

static float luma(const uint8_t* p) {
    return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
}

void cpu_grayscale(const uint8_t* src, uint8_t* dst, int width, int height) {
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            int i = (y * width + x) * 3;
            uint8_t g = clamp_u8(luma(src + i));
            dst[i] = dst[i + 1] = dst[i + 2] = g;
        }
}

static void make_kernel(float* k, int radius, float sigma) {
    int sz = 2 * radius + 1, n = sz * sz;
    float sum = 0;
    for (int y = -radius; y <= radius; y++)
        for (int x = -radius; x <= radius; x++) {
            float v = expf(-(float)(x*x + y*y) / (2.f * sigma * sigma));
            k[(y + radius) * sz + (x + radius)] = v;
            sum += v;
        }
    for (int i = 0; i < n; i++) k[i] /= sum;
}

void cpu_gaussian_blur(const uint8_t* src, uint8_t* dst, int width, int height, float sigma, int radius) {
    int sz = 2 * radius + 1;
    float k[17 * 17] = {};
    make_kernel(k, radius, sigma);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0, g = 0, b = 0;
            for (int ky = -radius; ky <= radius; ky++) {
                int iy = std::min(std::max(y + ky, 0), height - 1);
                for (int kx = -radius; kx <= radius; kx++) {
                    int ix = std::min(std::max(x + kx, 0), width - 1);
                    float w = k[(ky + radius) * sz + (kx + radius)];
                    const uint8_t* p = src + (iy * width + ix) * 3;
                    r += p[0] * w; g += p[1] * w; b += p[2] * w;
                }
            }
            int i = (y * width + x) * 3;
            dst[i] = clamp_u8(r); dst[i+1] = clamp_u8(g); dst[i+2] = clamp_u8(b);
        }
    }
}

void cpu_sobel(const uint8_t* src, uint8_t* dst, int width, int height) {
    auto px = [&](int y, int x) -> float {
        y = std::min(std::max(y, 0), height - 1);
        x = std::min(std::max(x, 0), width  - 1);
        return luma(src + (y * width + x) * 3);
    };
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float gx = -px(y-1,x-1) + px(y-1,x+1)
                       - 2.f*px(y,x-1) + 2.f*px(y,x+1)
                       - px(y+1,x-1) + px(y+1,x+1);
            float gy = -px(y-1,x-1) - 2.f*px(y-1,x) - px(y-1,x+1)
                       + px(y+1,x-1) + 2.f*px(y+1,x) + px(y+1,x+1);
            uint8_t m = clamp_u8(sqrtf(gx*gx + gy*gy));
            int i = (y * width + x) * 3;
            dst[i] = dst[i+1] = dst[i+2] = m;
        }
    }
}

void cpu_brightness_contrast(const uint8_t* src, uint8_t* dst, int width, int height, float brightness, float contrast) {
    int n = width * height * 3;
    for (int i = 0; i < n; i++)
        dst[i] = clamp_u8(src[i] * contrast + brightness);
}
