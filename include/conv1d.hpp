#ifndef CONV1D_HPP
#define CONV1D_HPP

#include "layer.hpp"
#include "tensor.hpp"
#include <memory>
#include <fstream>

class Conv1D : public Layer {
  friend class ResidualBlock;
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int dilation;
    bool use_weight_norm;

    Tensor weights;
    Tensor biases;

    std::unique_ptr<Tensor> input_cache; 
    
    Tensor grad_weights;
    Tensor grad_biases;

public:
    Conv1D(int in_channels, int out_channels, int kernel_size, int dilation, bool use_weight_norm = false);

    Conv1D(Conv1D&&) = default;
    Conv1D& operator=(Conv1D&&) = default;
    Conv1D(const Conv1D&) = delete;
    Conv1D& operator=(const Conv1D&) = delete;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
    
    void update(double learning_rate);
    void zero_grad();
    void clip_gradients(double threshold) override;
    
    void save(std::ofstream& out) const override;
    void load(std::ifstream& in) override;
};

#endif
