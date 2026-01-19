#ifndef CONV1D_HPP
#define CONV1D_HPP
#include "layer.hpp"
#include <memory>
class Conv1D : public Layer {
public:
    int in_channels, out_channels, kernel_size, dilation; bool use_weight_norm;
    Tensor weights, biases, input_cache, output_buffer, grad_input_buffer, grad_weights, grad_biases;
    Conv1D(int ic, int oc, int k, int d, bool wn=false);
    const Tensor& forward_ref(const Tensor& input);
    Tensor forward(const Tensor& input) override;
    const Tensor& backward(const Tensor& output_gradient) override;
    void update(double lr); void zero_grad(); void clip_gradients(double threshold) override;
    void save(std::ofstream& out) const override; void load(std::ifstream& in) override;
};
#endif
