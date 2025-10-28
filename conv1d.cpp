#include "conv1d.hpp"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <omp.h>

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
    std::mt19937 gen(42);
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
    input_cache = std::make_unique<Tensor>(input.clone());

    const int W_in = input.get_width();
    Tensor output(out_channels, W_in);
    
    const double* X = input.get_data();
    const double* W = weights.get_data();
    const double* B = biases.get_data();
    double* Y = output.get_data();

    const int p = (kernel_size - 1) * dilation;
    
    #pragma omp parallel for
    for (int o = 0; o < out_channels; ++o) {
        double bias = B[o];
        for (int t = 0; t < W_in; ++t) {
            double sum = bias;
            for (int i = 0; i < in_channels; ++i) {
                for (int k = 0; k < kernel_size; ++k) {
                    int input_t = t - (p - k * dilation);
                    if (input_t >= 0 && input_t < W_in) {
                        size_t x_idx = static_cast<size_t>(i) * W_in + input_t;
                        size_t w_idx = (static_cast<size_t>(o) * in_channels + i) * kernel_size + k;
                        sum += X[x_idx] * W[w_idx];
                    }
                }
            }
            Y[static_cast<size_t>(o) * W_in + t] = sum;
        }
    }
    return output;
}

Tensor Conv1D::backward(const Tensor& output_gradient) {
    if (!input_cache) {
        throw std::runtime_error("Backward called without forward pass.");
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

    #pragma omp parallel for
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

    #pragma omp parallel for
    for (int i = 0; i < in_channels; ++i) {
        for (int t = 0; t < W_in; ++t) {
             double grad_x_sum = 0.0;
             for (int o = 0; o < out_channels; ++o) {
                 for (int k = 0; k < kernel_size; ++k) {
                     int t_out = t + p - k * dilation;
                     if (t_out >= 0 && t_out < W_in) {
                         size_t y_idx = static_cast<size_t>(o) * W_in + t_out;
                         size_t w_idx = (static_cast<size_t>(o) * in_channels + i) * kernel_size + k;
                         grad_x_sum += grad_Y[y_idx] * W_data[w_idx];
                     }
                 }
             }
             grad_X[static_cast<size_t>(i) * W_in + t] = grad_x_sum;
        }
    }
    
    input_cache = nullptr;
    return grad_input;
}

void Conv1D::clip_gradients(double threshold) {
    double* gw_data = grad_weights.get_data();
    double* gb_data = grad_biases.get_data();
    size_t w_size = grad_weights.get_total_size();
    size_t b_size = grad_biases.get_total_size();

    #pragma omp parallel for
    for (size_t i = 0; i < w_size; ++i) {
        if (gw_data[i] > threshold) gw_data[i] = threshold;
        else if (gw_data[i] < -threshold) gw_data[i] = -threshold;
    }

    for (size_t i = 0; i < b_size; ++i) {
        if (gb_data[i] > threshold) gb_data[i] = threshold;
        else if (gb_data[i] < -threshold) gb_data[i] = -threshold;
    }
}


void Conv1D::update(double learning_rate) {
    double* w_data = weights.get_data();
    double* b_data = biases.get_data();
    const double* gw_data = grad_weights.get_data();
    const double* gb_data = grad_biases.get_data();
    
    size_t w_size = weights.get_total_size();
    size_t b_size = biases.get_total_size();

    #pragma omp parallel for
    for (size_t i = 0; i < w_size; ++i) {
        w_data[i] -= learning_rate * gw_data[i];
    }
    for (size_t i = 0; i < b_size; ++i) {
        b_data[i] -= learning_rate * gb_data[i];
    }
}

void Conv1D::save(std::ofstream& out) const {
    weights.save(out);
    biases.save(out);
}

void Conv1D::load(std::ifstream& in) {
    weights.load(in);
    biases.load(in);
}
