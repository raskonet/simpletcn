#ifndef RESIDUAL_BLOCK_HPP
#define RESIDUAL_BLOCK_HPP

#include "layer.hpp"
#include "conv1d.hpp"
#include "relu.hpp"
#include "dropout.hpp"
#include <memory>
#include <fstream>

class ResidualBlock : public Layer {
private:
    Conv1D conv1;
    ReLU relu1;
    Dropout dropout1;

    Conv1D conv2;
    ReLU relu2;
    Dropout dropout2;
    
    std::unique_ptr<Conv1D> downsample; 
    std::unique_ptr<Tensor> input_cache;

public:
    ResidualBlock(int in_channels, int n_filters, int kernel_size, int dilation, double dropout_rate);
    ~ResidualBlock() = default;

    ResidualBlock(ResidualBlock&&) = default;
    ResidualBlock& operator=(ResidualBlock&&) = default;
    ResidualBlock(const ResidualBlock&) = delete;
    ResidualBlock& operator=(const ResidualBlock&) = delete;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
    
    void update(double learning_rate);
    void zero_grad();
    void set_training_mode(bool training);
    void clip_gradients(double threshold) override;
    
    void save(std::ofstream& out) const override;
    void load(std::ifstream& in) override;
};

#endif
