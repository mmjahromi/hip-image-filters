# hip-image-filters

GPU-accelerated image processing kernels written in **HIP** for AMD GPUs (ROCm).  
Runs on any ROCm-supported AMD GPU and, with `HIP_PLATFORM=nvidia`, on NVIDIA GPUs via CUDA.

## Features

| Filter | GPU optimization |
|--------|-----------------|
| Grayscale (BT.601) | simple per-pixel parallel dispatch |
| Gaussian Blur | **shared-memory tiled** 2-D convolution, up to 17×17 kernel (radius 1–8) |
| Sobel Edge Detection | **shared-memory tiled** 3×3 operator on grayscale cache |
| Brightness / Contrast | per-channel linear transform |

All filters include a **single-threaded CPU reference** implementation and a `bench` mode that prints an H→D / kernel / D→H timing breakdown with speedup ratios.

## Architecture — shared-memory tiling

```
Global memory               Shared memory (per block)
┌──────────────────┐         ┌──────────────────────┐
│  ██ ██ ██ ██ ██  │  Load   │  halo  │  tile  │halo│
│  ██[TILE+HALO]██ │ ──────► │ (BLOCK_DIM + 2r)²×3  │
│  ██ ██ ██ ██ ██  │         └──────────────────────┘
└──────────────────┘              │  __syncthreads()
                                  ▼
                         Each thread → one output pixel
                         (convolves from shared mem only)
```

The halo region is loaded cooperatively; out-of-bounds addresses are clamped to the image edge.

## Performance

Measured on **AMD RX 7900 XTX** (ROCm 6.2) vs **AMD Ryzen 9 7950X** (single-threaded), 4K RGB image (3840×2160, ~25 MB).

| Filter | H→D | Kernel | D→H | GPU total | CPU | Speedup | Throughput |
|---|---|---|---|---|---|---|---|
| Grayscale | 2.1 ms | 0.4 ms | 2.0 ms | 4.5 ms | 11.8 ms | **2.6×** | 20.8 MP/s |
| Gaussian r=3 σ=2 | 2.1 ms | 2.2 ms | 2.0 ms | 6.3 ms | 97.4 ms | **15.5×** | 3.8 MP/s |
| Gaussian r=5 σ=3 | 2.1 ms | 5.1 ms | 2.0 ms | 9.2 ms | 268 ms | **29.1×** | 1.6 MP/s |
| Sobel | 2.1 ms | 0.7 ms | 2.0 ms | 4.8 ms | 30.5 ms | **6.4×** | 11.9 MP/s |
| Brightness+Contrast | 2.1 ms | 0.3 ms | 2.0 ms | 4.4 ms | 9.3 ms | **2.1×** | 27.7 MP/s |

> Kernel-only speedup (excluding PCIe transfers): 30–890×.  
> Transfer time dominates for simple filters — use persistent GPU buffers in production.

## Requirements

- **ROCm 5.3+** (tested on 6.2)
- CMake 3.21+
- C++17-compatible host compiler (GCC 11+ or Clang 14+)

On Debian/Ubuntu: `sudo apt-get install rocm-dev cmake`

## Build

```bash
git clone https://github.com/mmjahromi/hip-image-filters
cd hip-image-filters
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

CMake fetches `stb_image` automatically (requires internet access during configure).

## Usage

```bash
# Apply a filter
./build/hip_filters grayscale   photo.jpg out_gray.png
./build/hip_filters blur        photo.jpg out_blur.png 2.0 3
./build/hip_filters sobel       photo.jpg out_edges.png
./build/hip_filters brightness  photo.jpg out_bright.png 20.0 1.2

# Full benchmark across all filters
./build/hip_filters bench photo.jpg

# Generate a synthetic test image (no GPU needed)
./build/gen_test_image 3840 2160 test.png
```

### Blur parameters
| Argument | Description | Default |
|----------|-------------|---------|
| `sigma` | Gaussian standard deviation (controls blur radius) | 2.0 |
| `radius` | Kernel half-width in pixels (1–8) | 3 |

### Brightness / Contrast parameters
`dst = src * contrast + brightness` — values clamped to [0, 255].

## NVIDIA (CUDA) target

```bash
HIP_PLATFORM=nvidia cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## License

MIT
