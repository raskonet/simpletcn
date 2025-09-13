#ifndef TCN_HPP
#define TCN_HPP

#include "residual_block.hpp"
#include "tensor.hpp"
#include <vector>

class TCN {
private:
    std::vector<ResidualBlock> blocks;

public:
    TCN(int in_channels, int n_filters, int kernel_size, int levels, double dropout_rate = 0.2);
    ~TCN() = default;

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& output_gradient);
    
    void update(double learning_rate);
    void zero_grad();
    void set_training_mode(bool training);
};

#endif
