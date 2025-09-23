#include "conv1d.hpp"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>

Conv1D::Conv1D(int in_channels, int out_channels, int kernel_size, int dilation, bool use_weight_norm)
    : in_channels(in_channels),
      out_channels(out_channels),
      kernel_size(kernel_size),
      dilation(dilation),
      use_weight_norm(use_weight_norm),
      weights(out_channels, in_channels * kernel_size), 
      biases(out_channels, 1), 
      input_cache(nullptr),
      grad_weights(out_channels, in_channels * kernel_size),
      grad_biases(out_channels, 1)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    double std_dev = std::sqrt(2.0 / (in_channels * kernel_size));
    std::normal_distribution<> d(0, std_dev);

    double* w_data = weights.get_data();
    for (size_t i = 0; i < weights.get_total_size(); ++i) {
        w_data[i] = d(gen);
    }
    
    biases.zero();

    zero_grad();
}

void Conv1D::zero_grad() {
    grad_weights.zero();
    grad_biases.zero();
}

Tensor Conv1D::forward(const Tensor& input) {
    input_cache = &input;
    const int W_in = input.get_width();
    
    //The output width is the same as input width due to causal padding.
    Tensor output(out_channels, W_in);
    output.zero();

    const double* X = input.get_data();
    const double* W = weights.get_data();
    const double* B = biases.get_data();
    double* Y = output.get_data();

    // Padding 'p' is applied conceptually to the left of the sequence.
    const int p = (kernel_size - 1) * dilation;
    
    // Y[o][t] = B[o] + Σ_i Σ_k X_padded[i][t + k*d] * W[o][i][k]
    for (int o = 0; o < out_channels; ++o) {
        for (int t = 0; t < W_in; ++t) {
            double sum = B[o]; // Start with the bias for the output channel
            for (int i = 0; i < in_channels; ++i) {
                for (int k = 0; k < kernel_size; ++k) {
                    // Calculate the corresponding index in the unpadded input tensor
                    int input_t = t - (p - k * dilation);
                    
                    // If the index is valid (i.e., not in the padded region)
                    if (input_t >= 0 && input_t < W_in) {
                        // Index for X[i][input_t]
                        size_t x_idx = static_cast<size_t>(i) * W_in + input_t;
                        // Index for W[o][i][k]
                        size_t w_idx = (static_cast<size_t>(o) * in_channels + i) * kernel_size + k;
                        sum += X[x_idx] * W[w_idx];
                    }
                }
            }
            // Index for Y[o][t]
            Y[static_cast<size_t>(o) * W_in + t] = sum;
        }
    }
    return output;
}

Tensor Conv1D::backward(const Tensor& output_gradient) {
    if (!input_cache) {
        throw std::runtime_error("Backward called without forward pass. Input cache is null.");
    }
    
    const Tensor& X_cached = *input_cache;
    const int W_in = X_cached.get_width();

    const double* grad_Y = output_gradient.get_data();
    const double* X_data = X_cached.get_data();
    const double* W_data = weights.get_data();

    double* grad_W = grad_weights.get_data();
    double* grad_B = grad_biases.get_data();
    
    Tensor grad_input(in_channels, W_in);
    grad_input.zero();
    double* grad_X = grad_input.get_data();

    const int p = (kernel_size - 1) * dilation;

    // dL/dW[o][i][k] = Σ_t (dL/dY[o][t]) * X[i][t + k*d - p]
    // dL/dB[o] = Σ_t dL/dY[o][t]
    for (int o = 0; o < out_channels; ++o) {
        double bias_grad_sum = 0.0;
        for (int t = 0; t < W_in; ++t) {
            double grad_y_ot = grad_Y[static_cast<size_t>(o) * W_in + t];
            bias_grad_sum += grad_y_ot;
            
            for (int i = 0; i < in_channels; ++i) {
                for (int k = 0; k < kernel_size; ++k) {
                    int input_t = t - (p - k * dilation);
                    if (input_t >= 0 && input_t < W_in) {
                        size_t x_idx = static_cast<size_t>(i) * W_in + input_t;
                        size_t w_grad_idx = (static_cast<size_t>(o) * in_channels + i) * kernel_size + k;
                        grad_W[w_grad_idx] += X_data[x_idx] * grad_y_ot;
                    }
                }
            }
        }
        grad_B[o] += bias_grad_sum;
    }

    // dL/dX[i][t'] = Σ_o Σ_k (dL/dY[o][t' - k*d + p]) * W[o][i][k]
    // This is a "full" convolution. We iterate through the output gradient and "scatter" its influence back to the input gradient.
    for (int o = 0; o < out_channels; ++o) {
        for (int t = 0; t < W_in; ++t) {
            double grad_y_ot = grad_Y[static_cast<size_t>(o) * W_in + t];
            for (int i = 0; i < in_channels; ++i) {
                for (int k = 0; k < kernel_size; ++k) {
                    int input_t = t - (p - k * dilation);
                    if (input_t >= 0 && input_t < W_in) {
                        size_t x_grad_idx = static_cast<size_t>(i) * W_in + input_t;
                        size_t w_idx = (static_cast<size_t>(o) * in_channels + i) * kernel_size + k;
                        grad_X[x_grad_idx] += W_data[w_idx] * grad_y_ot;
                    }
                }
            }
        }
    }
    
    // Invalidate cache after use to prevent stale data in subsequent passes
    input_cache = nullptr;
    return grad_input;
}

void Conv1D::update(double learning_rate) {
    double* w_data = weights.get_data();
    double* b_data = biases.get_data();
    const double* gw_data = grad_weights.get_data();
    const double* gb_data = grad_biases.get_data();

    for (size_t i = 0; i < weights.get_total_size(); ++i) {
        w_data[i] -= learning_rate * gw_data[i];
    }
    for (size_t i = 0; i < biases.get_total_size(); ++i) {
        b_data[i] -= learning_rate * gb_data[i];
    }
}
