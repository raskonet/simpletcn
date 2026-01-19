#ifndef TCN_HPP
#define TCN_HPP

#include "residual_block.hpp"
#include "conv1d.hpp"
#include "tensor.hpp"
#include <vector>
#include <string>

class TCN {
private:
    std::vector<ResidualBlock> blocks;
    Conv1D final_conv; 

public:
    TCN(int in_channels, int n_filters, int kernel_size, int levels, int output_channels, double dropout_rate = 0.2);
    ~TCN() = default;

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& output_gradient);
    
    void update(double learning_rate);
    void zero_grad();
    void set_training_mode(bool training);
    void clip_gradients(double threshold);

    void save(const std::string& filename) const;
    void load(const std::string& filename);
};

#endif
