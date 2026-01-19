#include "relu.hpp"
#include "cuda_utils.hpp"
#include <algorithm>
const Tensor& ReLU::forward_ref(const Tensor& input) {
    input_cache.copy_from(input);
#ifdef USE_CUDA
    if (input_cache.get_device() == Device::GPU) { launch_relu(input_cache); return input_cache; }
#endif
    double* d = input_cache.get_data(); for(size_t i=0; i<input_cache.get_total_size(); ++i) d[i] = std::max(0.0, d[i]);
    return input_cache;
}
Tensor ReLU::forward(const Tensor& i) { forward_ref(i); return input_cache.clone(); }
const Tensor& ReLU::backward(const Tensor& g) {
    grad_input_buffer.reallocate(input_cache.get_channels(), input_cache.get_width());
#ifdef USE_CUDA
    if (g.get_device() == Device::GPU) {
        grad_input_buffer.to_device();
        launch_relu_backward(g, input_cache, grad_input_buffer);
        return grad_input_buffer;
    }
#endif
    Tensor go = g.clone(); go.to_host(); input_cache.to_host();
    double* p = go.get_data(); const double* x = input_cache.get_data(); double* d = grad_input_buffer.get_data();
    for(size_t i=0; i<go.get_total_size(); ++i) d[i] = (x[i]>0) ? p[i] : 0;
    return grad_input_buffer;
}
