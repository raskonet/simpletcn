#ifndef RELU_HPP
#define RELU_HPP

#include "layer.hpp"
#include "tensor.hpp"

class ReLU : public Layer {
private:
    const Tensor* input_cache; 

public:
    ReLU() : input_cache(nullptr) {}
    ~ReLU() = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
};
