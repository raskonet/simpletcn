#include "conv1d.hpp"
#include "cuda_utils.hpp"
#include <random>
#include <cmath>
#include <omp.h>

Conv1D::Conv1D(int ic, int oc, int k, int d, bool wn)
    : in_channels(ic), out_channels(oc), kernel_size(k), dilation(d), use_weight_norm(wn),
      weights(oc, ic*k), biases(oc, 1), grad_weights(oc, ic*k), grad_biases(oc, 1) {
    std::mt19937 gen(42); std::normal_distribution<> dist(0, std::sqrt(2.0/(ic*k)));
    double* w = weights.get_data(); for(size_t i=0; i<weights.get_total_size(); ++i) w[i] = dist(gen);
    biases.zero(); grad_weights.zero(); grad_biases.zero();
}
void Conv1D::zero_grad() { grad_weights.zero(); grad_biases.zero(); }
void Conv1D::update(double lr) {
#ifdef USE_CUDA
    if (weights.get_device() == Device::GPU) {
        launch_sgd_update(weights, grad_weights, lr);
        launch_sgd_update(biases, grad_biases, lr);
        return;
    }
#endif
    double* w = weights.get_data(); const double* gw = grad_weights.get_data();
    for(size_t i=0; i<weights.get_total_size(); ++i) w[i] -= lr * gw[i];
    double* b = biases.get_data(); const double* gb = grad_biases.get_data();
    for(size_t i=0; i<biases.get_total_size(); ++i) b[i] -= lr * gb[i];
}
const Tensor& Conv1D::forward_ref(const Tensor& input) {
#ifdef USE_CUDA
    if (input.get_device() == Device::GPU) {
        if (weights.get_device() == Device::CPU) weights.to_device();
        if (biases.get_device() == Device::CPU) biases.to_device();
        if (grad_weights.get_device() == Device::CPU) grad_weights.to_device();
        if (grad_biases.get_device() == Device::CPU) grad_biases.to_device();
        input_cache.copy_from(input); output_buffer.reallocate(out_channels, input.get_width());
        launch_conv1d(input_cache, weights, biases, output_buffer, dilation, kernel_size);
        return output_buffer;
    }
#endif
    // Optimization: avoid clone if possible by using input_cache as temporary in a full pipeline
    output_buffer = forward(input); 
    return output_buffer;
}
Tensor Conv1D::forward(const Tensor& input) {
    // Note: input.clone() is expensive. In high-perf loops, use forward_ref.
    input_cache = input.clone();
    Tensor output(out_channels, input.get_width());
    const double* X = input.get_data(); const double* W = weights.get_data(); const double* B = biases.get_data(); double* Y = output.get_data();
    int W_in = input.get_width(); int p = (kernel_size - 1) * dilation;
    #pragma omp parallel for
    for (int o = 0; o < out_channels; ++o) {
        double bias = B[o];
        for (int t = 0; t < W_in; ++t) {
            double sum = bias;
            for (int i = 0; i < in_channels; ++i) {
                for (int k = 0; k < kernel_size; ++k) {
                    int input_t = t - (p - k * dilation);
                    if (input_t >= 0 && input_t < W_in) sum += X[i*W_in+input_t] * W[(o*in_channels+i)*kernel_size+k];
                }
            }
            Y[o*W_in+t] = sum;
        }
    }
    return output;
}
const Tensor& Conv1D::backward(const Tensor& grad_output) {
    grad_input_buffer.reallocate(in_channels, input_cache.get_width());
    
#ifdef USE_CUDA
    if (grad_output.get_device() == Device::GPU) {
        grad_input_buffer.to_device();
        launch_conv1d_backward(grad_output, input_cache, weights, grad_input_buffer, grad_weights, grad_biases, dilation, kernel_size);
        return grad_input_buffer;
    }
#endif

    Tensor grad_cpu = grad_output.clone(); grad_cpu.to_host(); input_cache.to_host(); weights.to_host();
    const double* grad_Y = grad_cpu.get_data(); const double* X = input_cache.get_data(); const double* W = weights.get_data();
    grad_input_buffer.zero(); 
    double* dX = grad_input_buffer.get_data(); double* dW = grad_weights.get_data(); double* dB = grad_biases.get_data();
    int W_in = input_cache.get_width(); int p = (kernel_size - 1) * dilation;
    #pragma omp parallel for
    for(int o=0; o<out_channels; ++o) {
        double db_sum = 0;
        for(int t=0; t<W_in; ++t) {
            double g = grad_Y[o*W_in + t]; db_sum += g;
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
    return grad_input_buffer;
}
void Conv1D::clip_gradients(double t) {}
void Conv1D::save(std::ofstream& out) const { weights.save(out); biases.save(out); }
void Conv1D::load(std::ifstream& in) { weights.load(in); biases.load(in); }
