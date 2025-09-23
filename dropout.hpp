#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include "layer.hpp"
#include "tensor.hpp"
#include <memory> 

class Dropout : public Layer {
private:
    double dropout_rate;
    std::unique_ptr<Tensor> mask; 
    bool is_training;

public:
    Dropout(double rate);
    ~Dropout() = default;

    Dropout(const Dropout&) =delete;
    Dropout& operator=(const Dropout&) = delete;

    Dropout(Dropout&&) = default;
    Dropout& operator=(Dropout&&) = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;

    void set_training_mode(bool training);
};

#endif
