// Compares each GPU filter with its CPU reference on a small synthetic image.
// Allows a difference of 1 grey level (float rounding); Sobel gets 6, because the GPU kernel
// truncates the grey values to bytes before filtering (error bound about 5.7). Exit 1 on mismatch.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "filters.h"

static int check(const char* name, const std::vector<uint8_t>& gpu, const std::vector<uint8_t>& cpu, int tol) {
    int maxd = 0; size_t bad = 0;
    for (size_t i = 0; i < cpu.size(); i++) {
        int d = abs((int)gpu[i] - (int)cpu[i]);
        if (d > maxd) maxd = d;
        if (d > tol) bad++;
    }
    printf("%-20s max diff %d, %zu values off by more than %d: %s\n", name, maxd, bad, tol, bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}

int main() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) { printf("FAIL: no HIP device\n"); return 1; }
    const int w = 97, h = 61;   // not multiples of the tile size
    std::vector<uint8_t> src(w * h * 3), gpu(src.size()), cpu(src.size());
    for (size_t i = 0; i < src.size(); i++) src[i] = (uint8_t)((i * 37 + (i / 7) * 11) ^ (i >> 3));
    int fails = 0;
    auto run = [&](const char* name, auto gpu_fn, auto cpu_fn, int tol = 1) {
        std::fill(gpu.begin(), gpu.end(), 0xAA);   // sentinel: untouched output must fail
        gpu_fn(); cpu_fn();
        fails += check(name, gpu, cpu, tol);
    };
    run("grayscale", [&]{ gpu_grayscale(src.data(), gpu.data(), w, h); },
                     [&]{ cpu_grayscale(src.data(), cpu.data(), w, h); });
    run("gaussian r=3 s=2", [&]{ gpu_gaussian_blur(src.data(), gpu.data(), w, h, 2.f, 3); },
                            [&]{ cpu_gaussian_blur(src.data(), cpu.data(), w, h, 2.f, 3); });
    run("gaussian r=5 s=3", [&]{ gpu_gaussian_blur(src.data(), gpu.data(), w, h, 3.f, 5); },
                            [&]{ cpu_gaussian_blur(src.data(), cpu.data(), w, h, 3.f, 5); });
    run("sobel", [&]{ gpu_sobel(src.data(), gpu.data(), w, h); },
                 [&]{ cpu_sobel(src.data(), cpu.data(), w, h); }, 6);
    run("brightness+contrast", [&]{ gpu_brightness_contrast(src.data(), gpu.data(), w, h, 20.f, 1.2f); },
                               [&]{ cpu_brightness_contrast(src.data(), cpu.data(), w, h, 20.f, 1.2f); });
    printf(fails ? "FAILED\n" : "all filters match\n");
    return fails ? 1 : 0;
}
