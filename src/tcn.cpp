#include "tcn.hpp"
#include <stdexcept>
#include <iostream>
#include <fstream>

TCN::TCN(int in_channels, int n_filters, int kernel_size, int levels, int output_channels, double dropout_rate)
    : final_conv(n_filters, output_channels, 1, 1) 
{
    if (levels <= 0) {
        throw std::invalid_argument("Levels must be greater than 0.");
    }

    int current_in_channels = in_channels;
    int current_dilation = 1;

    for (int i = 0; i < levels; ++i) {
        blocks.emplace_back(
            current_in_channels, 
            n_filters, 
            kernel_size, 
            current_dilation, 
            dropout_rate
        );

        current_in_channels = n_filters;
        current_dilation *= 2; 
    }
}

Tensor TCN::forward(const Tensor& input) {
    Tensor x = blocks[0].forward(input);
    
    for (size_t i = 1; i < blocks.size(); ++i) {
        Tensor next_x = blocks[i].forward(x);
        x = std::move(next_x);
    }
    
    return final_conv.forward(x);
}

Tensor TCN::backward(const Tensor& output_gradient) {
    Tensor grad = final_conv.backward(output_gradient);
    
    for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
        Tensor next_grad = blocks[i].backward(grad);
        grad = std::move(next_grad);
    }
    
    return grad;
}

void TCN::clip_gradients(double threshold) {
    for (auto& block : blocks) {
        block.clip_gradients(threshold);
    }
    final_conv.clip_gradients(threshold);
}

void TCN::update(double learning_rate) {
    for (auto& block : blocks) {
        block.update(learning_rate);
    }
    final_conv.update(learning_rate);
}

void TCN::zero_grad() {
    for (auto& block : blocks) {
        block.zero_grad();
    }
    final_conv.zero_grad();
}

void TCN::set_training_mode(bool training) {
    for (auto& block : blocks) {
        block.set_training_mode(training);
    }
}

void TCN::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot open file for saving");
    
    for (const auto& block : blocks) {
        block.save(out);
    }
    final_conv.save(out);
    out.close();
    // std::cout << "[IO] Model saved to " << filename << std::endl;
}

void TCN::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file for loading");
    
    for (auto& block : blocks) {
        block.load(in);
    }
    final_conv.load(in);
    in.close();
    std::cout << "[IO] Model loaded from " << filename << std::endl;
}
