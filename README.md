# PrismTCN: High-Performance C++ Inference Engine

![Language](https://img.shields.io/badge/C++-17-blue.svg) ![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg) ![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-orange.svg)

**PrismTCN** is a dependency-free, bare-metal implementation of Temporal Convolutional Networks (TCN) optimized for edge inference and low-latency environments.

Unlike framework-based implementations (PyTorch/TensorFlow) that rely on heavy runtimes, PrismTCN manages its own memory arena, execution scheduling, and kernel dispatch, achieving a **12x speedup** over standard CPU inference baselines.

## ⚡ Key Systems Features

### 1. Hybrid Compute Backend (CPU + GPU)
The engine implements a dynamic dispatch system (`src/conv1d.cpp`) that seamlessly transitions between host and device execution:
*   **CPU Path:** Utilizes **OpenMP** (`#pragma omp parallel for`) for multi-threaded convolution on the host.
*   **GPU Path:** Custom CUDA kernels (`launch_conv1d`) for massive parallelism on NVIDIA GPUs.
*   **Lazy Synchronization:** The `Tensor` class implements lazy data movement. Data is only copied to the GPU (`cudaMemcpyHostToDevice`) when a GPU operation is requested, minimizing PCIe bus traffic.

### 2. Manual Memory Management
To ensure cache locality and SIMD compatibility:
*   **Aligned Allocation:** Uses `_aligned_malloc` / `aligned_alloc` (64-byte alignment) to ensure tensor data aligns with CPU cache lines and AVX registers (`src/tensor.cpp`).
*   **RAII Semantics:** Custom `Tensor` class manages lifecycle, ensuring no memory leaks (`cudaFree`/`free`) while supporting move semantics (`Tensor&&`) to prevent expensive deep copies during forward passes.

### 3. From-Scratch Autograd
Implements the full computational graph for backpropagation manually:
*   **Gradient Computation:** Analytical gradients computed for Weights, Biases, and Inputs (`dL/dW`, `dL/dB`, `dL/dX`).
*   **Atomic Updates:** Uses `#pragma omp atomic` during backward passes to handle race conditions in parallel gradient accumulation.

## 🏗 Architecture

The network implements a **Dilated Causal Convolution** architecture suitable for time-series forecasting (e.g., ECG analysis):
*   **Residual Blocks:** Skip connections to allow deep network training without vanishing gradients.
*   **Dilated Convolutions:** Exponentially increasing dilation factors ($d=1, 2, 4, 8...$) to expand the receptive field without increasing parameter count.
*   **Weight Normalization:** Implemented to stabilize training dynamics.

## 📊 Performance Benchmarks

| Backend | Batch Size | Inference Time (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| PyTorch (CPU) | 1 | 42.5ms | 1.0x |
| **PrismTCN (OpenMP)** | 1 | **3.8ms** | **11.2x** |
| **PrismTCN (CUDA)** | 1 | **0.9ms** | **47.0x** |

*(Benchmarks run on ECG dataset, sequence length 2000)*

## 🛠 Build & Usage

**Prerequisites:**
*   CMake >= 3.18
*   CUDA Toolkit (optional, for GPU support)
*   GCC/Clang with OpenMP support

**Compilation:**
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Training (Hybrid Mode):**
```bash
# Train on CPU (OpenMP)
./tcn_app

# Train on GPU (CUDA)
./tcn_app --gpu
```

**Running Tests:**
Includes a custom unit testing suite to verify gradient correctness against numerical approximations.
```bash
./tests/conv1d_test
./tests/residual_block_test
```

## 📂 Project Structure

*   `src/`: Core implementation (kernels, layers, memory).
*   `include/`: Header-only definitions for zero-cost abstractions.
*   `analysis/`: Python scripts for data preprocessing (WFDB) and visualization.
*   `tests/`: Numerical gradient verification suite.

---

