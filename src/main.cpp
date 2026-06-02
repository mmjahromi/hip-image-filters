#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include "stb_image.h"
#include "stb_image_write.h"
#include "filters.h"

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <filter> <input> <output> [options]\n\n"
        "Filters:\n"
        "  grayscale\n"
        "  blur <sigma> <radius>      (default: sigma=2.0 radius=3)\n"
        "  sobel\n"
        "  brightness <value> <contrast>  (default: 0.0 1.0)\n"
        "  bench                      run all filters and print timing table\n\n"
        "Examples:\n"
        "  %s blur input.png out.png 2.0 5\n"
        "  %s bench input.png /tmp/\n",
        prog, prog, prog);
}

static double cpu_now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

static void print_row(const char* name, int width, int height,
                      const TimingMs& gput, double cpu_ms)
{
    double total_gpu = gput.h2d + gput.kernel + gput.d2h;
    double mpix      = (double)width * height / 1e6;
    printf("  %-22s  %6.2f ms  %6.2f ms  %6.2f ms  %6.2f ms  %6.2f ms  %5.1f×  %6.1f MP/s\n",
           name,
           (double)gput.h2d,
           (double)gput.kernel,
           (double)gput.d2h,
           total_gpu,
           cpu_ms,
           cpu_ms / total_gpu,
           mpix / (gput.kernel / 1000.0));
}

static void run_bench(const uint8_t* img, int w, int h) {
    size_t sz = (size_t)w * h * 3;
    uint8_t* dst_gpu = new uint8_t[sz];
    uint8_t* dst_cpu = new uint8_t[sz];
    TimingMs t;
    double t0;

    printf("\n  GPU timings measured with hipEvent; CPU is single-threaded wall time.\n");
    printf("  Image: %d×%d  (%.1f MP)\n\n", w, h, (double)w*h/1e6);
    printf("  %-22s  %8s  %8s  %8s  %8s  %8s  %7s  %10s\n",
           "Filter", "H→D", "Kernel", "D→H", "GPU tot", "CPU", "Speedup", "Throughput");
    printf("  %s\n", std::string(98, '-').c_str());

    // Grayscale
    gpu_grayscale(img, dst_gpu, w, h, &t);
    t0 = cpu_now_ms(); cpu_grayscale(img, dst_cpu, w, h);
    print_row("grayscale", w, h, t, cpu_now_ms() - t0);

    // Gaussian blur r=3 sigma=2
    gpu_gaussian_blur(img, dst_gpu, w, h, 2.0f, 3, &t);
    t0 = cpu_now_ms(); cpu_gaussian_blur(img, dst_cpu, w, h, 2.0f, 3);
    print_row("gaussian r=3 σ=2", w, h, t, cpu_now_ms() - t0);

    // Gaussian blur r=5 sigma=3
    gpu_gaussian_blur(img, dst_gpu, w, h, 3.0f, 5, &t);
    t0 = cpu_now_ms(); cpu_gaussian_blur(img, dst_cpu, w, h, 3.0f, 5);
    print_row("gaussian r=5 σ=3", w, h, t, cpu_now_ms() - t0);

    // Sobel
    gpu_sobel(img, dst_gpu, w, h, &t);
    t0 = cpu_now_ms(); cpu_sobel(img, dst_cpu, w, h);
    print_row("sobel", w, h, t, cpu_now_ms() - t0);

    // Brightness/contrast
    gpu_brightness_contrast(img, dst_gpu, w, h, 20.f, 1.2f, &t);
    t0 = cpu_now_ms(); cpu_brightness_contrast(img, dst_cpu, w, h, 20.f, 1.2f);
    print_row("brightness+contrast", w, h, t, cpu_now_ms() - t0);

    printf("\n");
    delete[] dst_gpu;
    delete[] dst_cpu;
}

int main(int argc, char* argv[]) {
    if (argc < 3) { usage(argv[0]); return 1; }

    const char* filter  = argv[1];
    const char* in_path = argv[2];

    int w, h, ch;
    uint8_t* img = stbi_load(in_path, &w, &h, &ch, 3);
    if (!img) {
        fprintf(stderr, "Error: cannot load '%s'\n", in_path);
        return 1;
    }
    size_t sz = (size_t)w * h * 3;

    if (strcmp(filter, "bench") == 0) {
        run_bench(img, w, h);
        stbi_image_free(img);
        return 0;
    }

    if (argc < 4) { usage(argv[0]); stbi_image_free(img); return 1; }
    const char* out_path = argv[3];

    uint8_t* dst = new uint8_t[sz];
    TimingMs t;

    if (strcmp(filter, "grayscale") == 0) {
        gpu_grayscale(img, dst, w, h, &t);
    } else if (strcmp(filter, "blur") == 0) {
        float sigma  = argc > 4 ? atof(argv[4]) : 2.0f;
        int   radius = argc > 5 ? atoi(argv[5]) : 3;
        gpu_gaussian_blur(img, dst, w, h, sigma, radius, &t);
    } else if (strcmp(filter, "sobel") == 0) {
        gpu_sobel(img, dst, w, h, &t);
    } else if (strcmp(filter, "brightness") == 0) {
        float brightness = argc > 4 ? atof(argv[4]) : 0.f;
        float contrast   = argc > 5 ? atof(argv[5]) : 1.f;
        gpu_brightness_contrast(img, dst, w, h, brightness, contrast, &t);
    } else {
        fprintf(stderr, "Unknown filter: %s\n", filter);
        usage(argv[0]);
        stbi_image_free(img); delete[] dst; return 1;
    }

    stbi_write_png(out_path, w, h, 3, dst, w * 3);
    printf("Saved: %s  (%dx%d)\n", out_path, w, h);
    printf("  H→D: %.3f ms  kernel: %.3f ms  D→H: %.3f ms  total: %.3f ms\n",
           (double)t.h2d, (double)t.kernel, (double)t.d2h,
           (double)(t.h2d + t.kernel + t.d2h));

    stbi_image_free(img);
    delete[] dst;
    return 0;
}
