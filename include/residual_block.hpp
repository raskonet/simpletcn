#ifndef RESIDUAL_BLOCK_HPP
#define RESIDUAL_BLOCK_HPP

#include "layer.hpp"
#include "conv1d.hpp"
#include "relu.hpp"
#include "dropout.hpp"
#include <memory>

class ResidualBlockTester;

class ResidualBlock : public Layer {
private: 
    Conv1D conv1; 
    ReLU relu1; 
    Dropout dropout1;
    Conv1D conv2; 
    ReLU relu2; 
    Dropout dropout2;
    std::unique_ptr<Conv1D> downsample;
    Tensor output_buffer, grad_input_buffer; 

    friend class ResidualBlockTester;

public:
    ResidualBlock(int in_c, int n_fil, int k, int d, double drop);
    
    const Tensor& forward_ref(const Tensor& input);
    Tensor forward(const Tensor& input) override;
    const Tensor& backward(const Tensor& grad_out) override;
    
    void update(double lr); 
    void zero_grad(); 
    void set_training_mode(bool t);
    
    void save(std::ofstream& out) const override; 
    void load(std::ifstream& in) override;
};
#endif
