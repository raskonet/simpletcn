# TCN-CUDA: High Performance ECG Forecasting

A professional-grade implementation of Temporal Convolutional Networks using C++17 and CUDA for high-performance time-series analysis.

## 🏗 Architecture

The project is structured for production scalability:

*   **`src/`**: Core implementation logic (`.cpp` and `.cu`).
*   **`include/`**: Header definitions.
*   **`analysis/`**: Python dashboarding and preprocessing scripts.
*   **`tests/`**: Unit tests to verify gradient math and kernel logic.

## 🚀 Features

*   **Hybrid Backend**: Seamlessly transition between OpenMP (CPU) and CUDA (GPU).
*   **Zero-Copy Abstraction**: The `Tensor` class manages Host/Device synchronization automatically.
*   **Custom Kernels**: Hand-optimized CUDA kernels for Dilated Causal Convolutions.

## 🛠 Building the Project

Requirements: `cmake >= 3.18`, `cuda_toolkit`, `g++`.

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
