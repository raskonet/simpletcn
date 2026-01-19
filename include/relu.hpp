#ifndef RELU_HPP
#define RELU_HPP

#include "layer.hpp"
#include "tensor.hpp"
#include <memory>

class ReLU : public Layer {
private:
    std::unique_ptr<Tensor> input_cache; 

public:
    ReLU() : input_cache(nullptr) {}
    ~ReLU() = default;

    // --- FIX: Explicitly default move operations ---
    ReLU(ReLU&&) = default;
    ReLU& operator=(ReLU&&) = default;

    ReLU(const ReLU&) = delete;
    ReLU& operator=(const ReLU&) = delete;
    // -----------------------------------------------

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
};

#endif
