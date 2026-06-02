// Generates a synthetic test image (gradient + concentric rings) as PNG.
// No GPU required — useful for CI smoke tests.
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[]) {
    int w = argc > 1 ? atoi(argv[1]) : 1920;
    int h = argc > 2 ? atoi(argv[2]) : 1080;
    const char* out = argc > 3 ? argv[3] : "test_input.png";

    auto* img = new unsigned char[(size_t)w * h * 3];
    float cx = w * 0.5f, cy = h * 0.5f;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dx = x - cx, dy = y - cy;
            float r  = sqrtf(dx*dx + dy*dy);
            float ring = 0.5f + 0.5f * sinf(r * 0.05f);   // concentric rings

            unsigned char* p = img + (y * w + x) * 3;
            p[0] = (unsigned char)(255 * (float)x / w);
            p[1] = (unsigned char)(255 * ring);
            p[2] = (unsigned char)(255 * (float)y / h);
        }
    }

    stbi_write_png(out, w, h, 3, img, w * 3);
    printf("Generated %dx%d test image: %s\n", w, h, out);
    delete[] img;
}
