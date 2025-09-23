// File: relu_test.cpp

#include "relu.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Helper to compare tensors for testing purposes
bool tensors_are_equal(const Tensor& t1, const Tensor& t2, double tolerance = 1e-9) {
    if (t1.get_total_size() != t2.get_total_size()) return false;
    const double* d1 = t1.get_data();
    const double* d2 = t2.get_data();
    for (size_t i = 0; i < t1.get_total_size(); ++i) {
        if (std::abs(d1[i] - d2[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << d1[i] << " != " << d2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void test_relu_layer() {
    std::cout << "--- Running ReLU Layer Test ---" << std::endl;

    // --- 1. SETUP ---
    ReLU layer;
    const int channels = 1;
    const int width = 7;
    
    Tensor input(channels, width);
    std::vector<double> input_vals = {-3.0, -2.5, -0.1, 0.0, 0.5, 1.0, 4.2};
    for(int i = 0; i < width; ++i) input.get_data()[i] = input_vals[i];

    // --- 2. TEST FORWARD PASS ---
    std::cout << "Testing forward pass..." << std::endl;
    Tensor output = layer.forward(input);

    Tensor expected_output(channels, width);
    std::vector<double> expected_output_vals = {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 4.2};
    for(int i = 0; i < width; ++i) expected_output.get_data()[i] = expected_output_vals[i];

    assert(tensors_are_equal(output, expected_output));
    std::cout << "PASS: Forward pass is correct." << std::endl;

    // --- 3. TEST BACKWARD PASS ---
    std::cout << "\nTesting backward pass..." << std::endl;
    Tensor grad_output(channels, width);
    std::vector<double> grad_output_vals = {10, 20, 30, 40, 50, 60, 70};
    for(int i = 0; i < width; ++i) grad_output.get_data()[i] = grad_output_vals[i];

    Tensor grad_input = layer.backward(grad_output);

    Tensor expected_grad_input(channels, width);
    // Gradient should be 0 where input was <= 0, and passed through otherwise.
    std::vector<double> expected_grad_input_vals = {0.0, 0.0, 0.0, 0.0, 50, 60, 70};
    for(int i = 0; i < width; ++i) expected_grad_input.get_data()[i] = expected_grad_input_vals[i];

    assert(tensors_are_equal(grad_input, expected_grad_input));
    std::cout << "PASS: Backward pass is correct." << std::endl;
}

int main() {
    try {
        test_relu_layer();
        std::cout << "\n[SUCCESS] All ReLU tests passed." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILURE] A test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
