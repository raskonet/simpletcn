````md
# PrismTCN: High-Performance Hybrid Inference Engine

![Language](https://img.shields.io/badge/C++-17-blue.svg) ![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg) ![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-orange.svg)

**PrismTCN** is a bare-metal, dependency-free implementation of Temporal Convolutional Networks (TCN) designed for **latency-critical** time-series forecasting.

Unlike standard frameworks (PyTorch/TensorFlow) which incur significant interpreter and dispatcher overhead for small batch sizes, PrismTCN runs on a custom C++ runtime with manual memory management and a hybrid CPU/GPU backend.

---

## ⚡ Key Systems Optimizations

### 1. Hybrid Compute Backend (Dynamic Dispatch)
The engine detects available hardware at runtime and dispatches kernels accordingly:
* **CPU Path:** Utilizes **OpenMP** (`#pragma omp parallel for`) with AVX-friendly memory layouts for multi-threaded host execution.
* **GPU Path:** Custom **CUDA Kernels** (`src/cuda_ops.cu`) for massive parallelism. Implements a custom `prism_atomic_add` wrapper to ensure compatibility across Pascal and Volta architectures where `atomicAdd` for doubles was inconsistent.

### 2. Zero-Copy Memory Management
To minimize the "OS Tax" of memory allocation:
* **Lazy Synchronization:** The `Tensor` class implements a lazy `to_host()` / `to_device()` protocol. Data is only moved across the PCIe bus when strictly necessary for a compute operation.
* **Smart Reallocation:** The `reallocate()` method (`src/tensor.cpp`) checks capacity before asking the OS for new memory, preventing expensive `malloc`/`cudaMalloc` calls during the hot inference loop.
* **Aligned Storage:** Enforces 64-byte memory alignment (`_aligned_malloc`) to prevent false sharing and ensure cache line alignment.

### 3. From-Scratch Autograd
Implements the full computational graph for backpropagation without an external DAG engine:
* **Manual Gradient Derivation:** Analytical gradients computed for Dilated Convolutions and Residual Blocks.
* **Thread-Safe Accumulation:** Uses atomic operations during the backward pass to handle race conditions when multiple output neurons contribute to the same weight gradient.

---

## 📊 Performance Benchmarks

Benchmarks run on **NVIDIA T4 GPU**.  
Sequence Length: **2000** · Batch Size: **1** (Real-Time / Streaming Inference)

| Implementation | Backend | Avg Inference Latency | Speedup vs PyTorch CPU |
| :--- | :--- | :--- | :--- |
| PyTorch (Standard) | CPU | 2.5463 ms | 1.0× |
| PyTorch (Standard) | CUDA | 1.4710 ms | 1.73× |
| **PrismTCN** | **OpenMP (C++)** | **1.5806 ms** | **1.61×** |
| **PrismTCN** | **CUDA (C++)** | **1.1342 ms** | **2.24×** |

### Observations
* **C++ CPU beats PyTorch CPU**, validating the removal of Python dispatch, interpreter overhead, and dynamic graph costs.
* **CUDA speedup is intentionally modest** at Batch Size = 1, where kernel launch latency and synchronization dominate.
* **PrismTCN CUDA outperforms PyTorch CUDA** even in a latency-bound regime, indicating tighter kernel fusion and reduced framework overhead.
* These benchmarks reflect **true online inference**, not throughput-optimized batching.

> **Note:** At Batch Size = 1, performance is latency-bound rather than FLOP-bound. Substantial GPU gains emerge at higher batch sizes or longer sequences.

---

## 🏗 Architecture

The model implements a **Dilated Causal Convolution** architecture:
* **Causal Padding:** Ensures no leakage of future information into the past (critical for financial and medical forecasting).
* **Exponential Dilation:** Dilation factors (`d = 2^i`) allow the receptive field to grow exponentially with network depth.
* **Residual Connections:** `1×1` convolutions used for channel projection in skip connections.

---

## 🛠 Build & Usage

### Prerequisites
* CMake ≥ 3.18
* GCC or Clang with OpenMP support
* CUDA Toolkit (optional, auto-detected)

### Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
````

### Run Training

```bash
# Train using OpenMP backend
./tcn_app

# Train using CUDA backend
./tcn_app --gpu
```

### Run Inference Benchmark

```bash
./tcn_app --inference
./tcn_app --inference --gpu
```

---

## 🧪 Testing & Verification

The project includes a rigorous testing suite (`tests/`) to ensure numerical stability and correctness:

* **`conv1d_test`** — Verifies analytical gradients against numerical gradients (finite differences).
* **`cuda_test`** — Confirms CPU and GPU kernels match within a tolerance of `1e-5`.
* **`tcn_test`** — Enforces strict causality (future inputs never influence past outputs).

---

## 📂 Project Structure

```text
├── include/        # Header-only abstractions (RAII Tensor, Layers)
├── src/
│   ├── cuda_ops.cu # Custom CUDA Kernels (Conv1d, ReLU, Dropout)
│   ├── tensor.cpp  # Memory Management & Aligned Allocation
│   └── tcn.cpp     # Network Graph Construction
├── tests/          # Numerical Gradient Verification
└── analysis/       # Python scripts for visualization & data loading
```

---

## 📌 Reproducibility Notes

* All benchmarks were averaged over multiple runs after warm-up.
* CPU measurements were taken with OpenMP enabled.
* GPU benchmarks include kernel launch and synchronization overhead.
* Python benchmarks used eager-mode PyTorch with identical model structure.

---

```
```

