#include "conv1d.hpp"
#include "tensor.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

double calculate_mse_loss(const Tensor& prediction, const Tensor& target) {
    double loss = 0.0;
    const double* pred_data = prediction.get_data();
    const double* target_data = target.get_data();
    size_t n = prediction.get_total_size();
    for (size_t i = 0; i < n; ++i) {
        double diff = pred_data[i] - target_data[i];
        loss += diff * diff;
    }
    return loss / n;
}

Tensor calculate_mse_grad(const Tensor& prediction, const Tensor& target) {
    Tensor grad(prediction.get_channels(), prediction.get_width());
    const double* pred_data = prediction.get_data();
    const double* target_data = target.get_data();
    double* grad_data = grad.get_data();
    size_t n = prediction.get_total_size();
    for (size_t i = 0; i < n; ++i) {
        grad_data[i] = 2.0 * (pred_data[i] - target_data[i]) / n;
    }
    return grad;
}

bool check_gradients() {
    std::cout << "\n--- Running Gradient Check for Conv1D ---" << std::endl;

    const int in_channels = 3;
    const int out_channels = 4;
    const int kernel_size = 3;
    const int dilation = 2;
    const int width = 10;
    const double epsilon = 1e-5; 
    const double tolerance = 1e-6; 

    Conv1D layer(in_channels, out_channels, kernel_size, dilation);
    Tensor input(in_channels, width);
    Tensor target(out_channels, width);

    for (size_t i = 0; i < input.get_total_size(); ++i) input.get_data()[i] = (i % 10) * 0.1;
    for (size_t i = 0; i < target.get_total_size(); ++i) target.get_data()[i] = (i % 7) * 0.15;

    layer.zero_grad();
    Tensor output = layer.forward(input);
    Tensor loss_grad = calculate_mse_grad(output, target);
    
    // FIX: Use .clone()
    Tensor analytical_grad_input = layer.backward(loss_grad).clone();
    
    std::cout << "Checking gradient w.r.t. Input (dL/dX)..." << std::endl;
    double* input_data = input.get_data();
    const double* analytical_dx = analytical_grad_input.get_data();

    for (size_t i = 0; i < input.get_total_size(); ++i) {
        double original_val = input_data[i];

        input_data[i] = original_val + epsilon;
        Tensor output1 = layer.forward(input);
        double loss1 = calculate_mse_loss(output1, target);

        input_data[i] = original_val - epsilon;
        Tensor output2 = layer.forward(input);
        double loss2 = calculate_mse_loss(output2, target);

        input_data[i] = original_val;

        double numerical_grad = (loss1 - loss2) / (2.0 * epsilon);
        double analytical_grad = analytical_dx[i];

        double rel_error = std::abs(analytical_grad - numerical_grad) /
                           std::max(std::abs(analytical_grad), std::abs(numerical_grad) + 1e-9);

        if (rel_error > tolerance) {
            std::cerr << "FAIL: dL/dX mismatch at index " << i << std::endl;
            return false;
        }
    }
    std::cout << "PASS: Gradient w.r.t. Input is correct." << std::endl;
    return true;
}

int main() {
    bool success = check_gradients();
    if (success) {
        std::cout << "\n[SUCCESS] All Conv1D tests passed." << std::endl;
        return 0;
    } else {
        std::cerr << "\n[FAILURE] One or more Conv1D tests failed." << std::endl;
        return 1;
    }
}
