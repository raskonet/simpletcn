#include "dropout.hpp"
#include "cuda_utils.hpp"
#include <random>
const Tensor& Dropout::forward_ref(const Tensor& input) {
    if(!is_training || rate==0.0) return input;
    output_buf.reallocate(input.get_channels(), input.get_width());
    mask.reallocate(input.get_channels(), input.get_width());
    mask.to_host(); double* m = mask.get_data();
    static std::mt19937 gen(42); std::bernoulli_distribution d(1.0-rate);
    for(size_t i=0; i<mask.get_total_size(); ++i) m[i] = d(gen)?1.0:0.0;
    double s = 1.0/(1.0-rate);
    #ifdef USE_CUDA
    if(input.get_device()==Device::GPU) { mask.to_device(); launch_dropout_apply(input, mask, output_buf, s); return output_buf; }
    #endif
    const double* in = input.get_data(); double* out = output_buf.get_data();
    for(size_t i=0; i<input.get_total_size(); ++i) out[i] = in[i] * m[i] * s; 
    return output_buf;
}
Tensor Dropout::forward(const Tensor& i) { return forward_ref(i).clone(); }
const Tensor& Dropout::backward(const Tensor& g) {
    if(!is_training || rate==0.0) return g;
    grad_input_buffer.copy_from(g); 
    
    grad_input_buffer.to_host(); mask.to_host();
    double* d = grad_input_buffer.get_data(); const double* m = mask.get_data(); double s = 1.0/(1.0-rate);
    for(size_t i=0; i<grad_input_buffer.get_total_size(); ++i) d[i] *= m[i] * s;
    
    #ifdef USE_CUDA
    if(g.get_device()==Device::GPU) grad_input_buffer.to_device(); 
    #endif
    return grad_input_buffer;
}
