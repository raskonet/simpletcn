#ifndef DROPOUT_HPP
#define DROPOUT_HPP
#include "layer.hpp"
class Dropout : public Layer {
    double rate; bool is_training; Tensor mask, output_buf, grad_input_buffer;
public:
    Dropout(double r) : rate(r), is_training(true) {}
    void set_training_mode(bool t) { is_training = t; }
    const Tensor& forward_ref(const Tensor& input);
    Tensor forward(const Tensor& input) override;
    const Tensor& backward(const Tensor& g) override;
};
#endif
