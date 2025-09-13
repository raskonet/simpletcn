#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output_gradient) = 0;
};


#endif
