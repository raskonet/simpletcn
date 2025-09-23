#ifndef CONV1D_HPP
#define CONV1D_HPP

#include "layer.hpp"
#include "tensor.hpp"
#include <memory>

void test_numerical_gradients();

class Conv1D : public Layer {
  friend void test_numerical_gradients();
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int dilation;
    int padding;
    bool use_weight_norm;

    Tensor weights;
    Tensor biases;

    std::unique_ptr<Tensor> g; 

    const Tensor* input_cache; 
    
    Tensor grad_weights;
    Tensor grad_biases;
    std::unique_ptr<Tensor> grad_g;

public:
    Conv1D(int in_channels, int out_channels, int kernel_size, int dilation, bool use_weight_norm = false);

    Conv1D(const Conv1D&) = delete;
    Conv1D& operator=(const Conv1D&) = delete;
    
    // Move constructor for emplacement in std::vector
    Conv1D(Conv1D&&) = default;
    Conv1D& operator=(Conv1D&&) = default;


    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
    
    void update(double learning_rate);
    void zero_grad();
};

#endif
