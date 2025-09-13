#ifndef RESIDUAL_BLOCK_HPP
#define RESIDUAL_BLOCK_HPP

#include "layer.hpp"
#include "conv1d.hpp"
#include "relu.hpp"
#include "dropout.hpp"
#include <memory>

class ResidualBlock : public Layer {
private:
    Conv1D conv1;
    ReLU relu1;
    Dropout dropout1;

    Conv1D conv2;
    ReLU relu2;
    Dropout dropout2;
    
    std::unique_ptr<Conv1D> downsample; 

public:
    ResidualBlock(int in_channels, int n_filters, int kernel_size, int dilation, double dropout_rate);
    ~ResidualBlock() = default;

    ResidualBlock(const ResidualBlock&) = delete;
    ResidualBlock& operator=(const ResidualBlock&) = delete;

    ResidualBlock(ResidualBlock&&) = default;
    ResidualBlock& operator=(ResidualBlock&&) = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
    
    void update(double learning_rate);
    void zero_grad();
    void set_training_mode(bool training);
};

#endif
