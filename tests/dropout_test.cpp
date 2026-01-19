#include "dropout.hpp"
#include "tensor.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <cassert>

// Simple helper to check if two tensors are identical
bool tensors_are_equal(const Tensor& t1, const Tensor& t2) {
    if (t1.get_total_size() != t2.get_total_size()) return false;
    const double* d1 = t1.get_data();
    const double* d2 = t2.get_data();
    for (size_t i = 0; i < t1.get_total_size(); ++i) {
        if (std::abs(d1[i] - d2[i]) > 1e-9) return false;
    }
    return true;
}

// Test 1: Inference mode should be an identity function
void test_inference_mode() {
    std::cout << "--- Test 1: Inference Mode ---" << std::endl;
    Dropout layer(0.5);
    layer.set_training_mode(false);

    Tensor input(3, 100);
    for (size_t i = 0; i < input.get_total_size(); ++i) {
        input.get_data()[i] = static_cast<double>(i);
    }

    Tensor output = layer.forward(input);
    assert(tensors_are_equal(input, output));
    std::cout << "PASS: Forward pass in inference mode is an identity function." << std::endl;

    Tensor grad_output(3, 100);
     for (size_t i = 0; i < grad_output.get_total_size(); ++i) {
        grad_output.get_data()[i] = static_cast<double>(i) * 0.5;
    }
    
    // FIX: Use .clone() because backward returns a reference, and we want a new object
    Tensor grad_input = layer.backward(grad_output).clone();
    
    assert(tensors_are_equal(grad_output, grad_input));
    std::cout << "PASS: Backward pass in inference mode is an identity function." << std::endl;
}

// Test 2: Training mode statistical properties
void test_training_mode_stats() {
    std::cout << "\n--- Test 2: Training Mode Statistics ---" << std::endl;
    double rate = 0.4;
    Dropout layer(rate);
    layer.set_training_mode(true);

    const int num_elements = 10000;
    Tensor input(1, num_elements);
    for(int i = 0; i < num_elements; ++i) input.get_data()[i] = 1.0;

    Tensor output = layer.forward(input);
    
    int zero_count = 0;
    double sum = 0.0;
    const double scale = 1.0 / (1.0 - rate);
    const double* out_data = output.get_data();

    for (int i = 0; i < num_elements; ++i) {
        sum += out_data[i];
        if (std::abs(out_data[i]) < 1e-9) {
            zero_count++;
        } else {
            assert(std::abs(out_data[i] - scale) < 1e-9);
        }
    }
    
    double zero_fraction = static_cast<double>(zero_count) / num_elements;
    double mean = sum / num_elements;

    std::cout << "Dropout Rate: " << rate << std::endl;
    std::cout << "Observed zero fraction: " << zero_fraction << " (Expected ~" << rate << ")" << std::endl;
    std::cout << "Observed mean: " << mean << " (Expected ~1.0)" << std::endl;

    assert(std::abs(zero_fraction - rate) < 0.02);
    assert(std::abs(mean - 1.0) < 0.05);

    std::cout << "PASS: Statistical properties (zero fraction and mean) are correct." << std::endl;
}

// Test 3: Backward pass in training mode
void test_training_backward() {
    std::cout << "\n--- Test 3: Training Mode Backward Pass ---" << std::endl;
    Dropout layer(0.5);
    layer.set_training_mode(true);
    
    Tensor input(1, 10);
    for(int i=0; i<10; ++i) input.get_data()[i] = 1.0;
    
    // Run forward to generate an internal mask
    Tensor output = layer.forward(input);

    // Create a gradient to backpropagate
    Tensor grad_output(1, 10);
    for(int i=0; i<10; ++i) grad_output.get_data()[i] = 2.0;
    
    // FIX: Use .clone()
    Tensor grad_input = layer.backward(grad_output).clone();
    
    const double* out_data = output.get_data();
    const double* gi_data = grad_input.get_data();
    const double scale = 1.0 / (1.0 - 0.5);

    for (int i=0; i<10; ++i) {
        if (std::abs(out_data[i]) < 1e-9) { 
            assert(std::abs(gi_data[i]) < 1e-9); 
        } else { 
            assert(std::abs(gi_data[i] - 2.0 * scale) < 1e-9); 
        }
    }
    std::cout << "PASS: Backward pass correctly applies the cached mask and scaling." << std::endl;
}

int main() {
    try {
        test_inference_mode();
        test_training_mode_stats();
        test_training_backward();
        std::cout << "\n[SUCCESS] All Dropout tests passed." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILURE] A test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
