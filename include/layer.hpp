#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"
#include <fstream>

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output_gradient) = 0;
    virtual void clip_gradients(double threshold) {}
    virtual void save(std::ofstream& out) const {}
    virtual void load(std::ifstream& in) {}
};

#endif
