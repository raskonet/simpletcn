#include "conv1d.hpp"
#include "cuda_utils.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <omp.h>

Conv1D::Conv1D(int in_channels, int out_channels, int kernel_size, int dilation, bool use_weight_norm)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), dilation(dilation),
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
    double* w = weights.get_data();
    for (size_t i = 0; i < weights.get_total_size(); ++i) w[i] = d(gen);
    biases.zero();
    grad_weights.zero();
    grad_biases.zero();
}

void Conv1D::zero_grad() {
    grad_weights.zero();
    grad_biases.zero();
}

void Conv1D::clip_gradients(double threshold) {
    // CPU clip (simplified)
    double* gw = grad_weights.get_data();
    double* gb = grad_biases.get_data();
    for(size_t i=0; i<grad_weights.get_total_size(); ++i) {
        if(gw[i] > threshold) gw[i] = threshold;
        if(gw[i] < -threshold) gw[i] = -threshold;
    }
    for(size_t i=0; i<grad_biases.get_total_size(); ++i) {
        if(gb[i] > threshold) gb[i] = threshold;
        if(gb[i] < -threshold) gb[i] = -threshold;
    }
}

void Conv1D::update(double lr) {
    double* w = weights.get_data();
    const double* gw = grad_weights.get_data();
    for(size_t i=0; i<weights.get_total_size(); ++i) w[i] -= lr * gw[i];
    
    double* b = biases.get_data();
    const double* gb = grad_biases.get_data();
    for(size_t i=0; i<biases.get_total_size(); ++i) b[i] -= lr * gb[i];
}

Tensor Conv1D::forward(const Tensor& input) {
    if (input.get_device() == Device::GPU) {
        // --- GPU PATH ---
        // Lazy load weights to GPU if needed
        if (weights.get_device() == Device::CPU) weights.to_device();
        if (biases.get_device() == Device::CPU) biases.to_device();

        input_cache = std::make_unique<Tensor>(input.clone());
        Tensor output(out_channels, input.get_width());
        output.to_device(); 

        launch_conv1d(input, weights, biases, output, dilation, kernel_size);
        return output;
    } 
    else {
        // --- CPU PATH ---
        input_cache = std::make_unique<Tensor>(input.clone());
        Tensor output(out_channels, input.get_width());
        const double* X = input.get_data();
        const double* W = weights.get_data();
        const double* B = biases.get_data();
        double* Y = output.get_data();
        int W_in = input.get_width();
        int p = (kernel_size - 1) * dilation;

        #pragma omp parallel for
        for (int o = 0; o < out_channels; ++o) {
            double bias = B[o];
            for (int t = 0; t < W_in; ++t) {
                double sum = bias;
                for (int i = 0; i < in_channels; ++i) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int input_t = t - (p - k * dilation);
                        if (input_t >= 0 && input_t < W_in) {
                            sum += X[i * W_in + input_t] * W[(o * in_channels + i) * kernel_size + k];
                        }
                    }
                }
                Y[o * W_in + t] = sum;
            }
        }
        return output;
    }
}

Tensor Conv1D::backward(const Tensor& grad_output) {
    // Fallback to CPU for backward pass in this version
    Tensor cpu_grad = grad_output.clone();
    cpu_grad.to_host(); 
    
    if(input_cache->get_device() == Device::GPU) input_cache->to_host();

    const double* grad_Y = cpu_grad.get_data();
    const double* X = input_cache->get_data();
    // Ensure weights/grads are on CPU for update
    weights.to_host();
    const double* W = weights.get_data(); 
    
    Tensor grad_input(in_channels, input_cache->get_width());
    grad_input.zero();
    double* dX = grad_input.get_data();
    double* dW = grad_weights.get_data();
    double* dB = grad_biases.get_data();
    int W_in = input_cache->get_width();
    int p = (kernel_size - 1) * dilation;

    #pragma omp parallel for
    for(int o=0; o<out_channels; ++o) {
        double db_sum = 0;
        for(int t=0; t<W_in; ++t) {
            double g = grad_Y[o*W_in + t];
            db_sum += g;
            for(int i=0; i<in_channels; ++i) {
                for(int k=0; k<kernel_size; ++k) {
                    int in_t = t - (p - k * dilation);
                    if(in_t >= 0 && in_t < W_in) {
                        #pragma omp atomic
                        dW[(o*in_channels + i)*kernel_size + k] += X[i*W_in + in_t] * g;
                        
                        #pragma omp atomic
                        dX[i*W_in + in_t] += g * W[(o*in_channels + i)*kernel_size + k];
                    }
                }
            }
        }
        dB[o] += db_sum;
    }
    input_cache = nullptr;
    return grad_input;
}

void Conv1D::save(std::ofstream& out) const { weights.save(out); biases.save(out); }
void Conv1D::load(std::ifstream& in) { weights.load(in); biases.load(in); }
