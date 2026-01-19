#ifndef RELU_HPP
#define RELU_HPP
#include "layer.hpp"
class ReLU : public Layer {
    Tensor input_cache, grad_input_buffer;
public:
    Tensor forward(const Tensor& input) override;
    const Tensor& forward_ref(const Tensor& input);
    const Tensor& backward(const Tensor& g) override;
};
#endif
