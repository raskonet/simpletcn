#include "residual_block.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cmath>
#include <algorithm>

// --- Helpers ---
double calculate_total_loss(const Tensor& prediction) {
    double loss = 0.0;
    const double* pred_data = prediction.get_data();
    for (size_t i = 0; i < prediction.get_total_size(); ++i) {
        loss += pred_data[i] * pred_data[i];
    }
    return loss / 2.0;
}

Tensor calculate_loss_grad(const Tensor& prediction) {
    Tensor grad(prediction.get_channels(), prediction.get_width());
    const double* pred_data = prediction.get_data();
    double* grad_data = grad.get_data();
    for (size_t i = 0; i < prediction.get_total_size(); ++i) {
        grad_data[i] = pred_data[i];
    }
    return grad;
}

// --- The Friend Class ---
class ResidualBlockTester {
public:
    static void run_numerical_check() {
        std::cout << "\n--- Test 3: Numerical Gradient Check (Minute Details) ---" << std::endl;
        const int in_channels = 2;
        const int n_filters = 3;
        const int width = 5;
        const int kernel_size = 2;
        const int dilation = 1;
        const double epsilon = 1e-5;
        const double tolerance = 1e-4;

        for (bool use_downsample : {false, true}) {
            int current_in = use_downsample ? in_channels : n_filters;
            std::cout << "  Testing with in_channels=" << current_in << ", n_filters=" << n_filters 
                      << " (downsample: " << std::boolalpha << use_downsample << ")" << std::endl;

            ResidualBlock block(current_in, n_filters, kernel_size, dilation, 0.0);
            block.set_training_mode(false);

            Tensor input(current_in, width);
            for(size_t i = 0; i < input.get_total_size(); ++i) input.get_data()[i] = (i % 5) * 0.1 + 0.1;

            // 1. Analytical Gradient
            block.zero_grad();
            Tensor output = block.forward(input);
            Tensor loss_grad = calculate_loss_grad(output);
            
            // Note: We ignore return value to avoid copy-constructor error, 
            // the side effects (grad_weights update) are what we care about here.
            block.backward(loss_grad);

            // 2. Numerical Check
            // We can access private members (conv1, conv2) because we are a friend class
            auto check_layer = [&](Conv1D& layer, const std::string& name) {
                double* weights = layer.weights.get_data();
                const double* analytical_grads = layer.grad_weights.get_data();
                
                size_t check_count = std::min(layer.weights.get_total_size(), size_t(10));
                
                for (size_t i = 0; i < check_count; ++i) {
                    double original_weight = weights[i];
                    
                    weights[i] = original_weight + epsilon;
                    double loss1 = calculate_total_loss(block.forward(input));

                    weights[i] = original_weight - epsilon;
                    double loss2 = calculate_total_loss(block.forward(input));

                    weights[i] = original_weight; // Restore

                    double numerical_grad = (loss1 - loss2) / (2.0 * epsilon);
                    double analytical_grad = analytical_grads[i];

                    double numerator = std::abs(analytical_grad - numerical_grad);
                    double denominator = std::max({std::abs(analytical_grad), std::abs(numerical_grad), 1e-9});
                    double rel_error = numerator / denominator;

                    if (rel_error > tolerance) {
                        std::cerr << "FAIL: High relative error in " << name << " at weight " << i << std::endl;
                        std::cerr << "  Analytical: " << analytical_grad << ", Numerical: " << numerical_grad
                                  << ", Rel. Error: " << rel_error << std::endl;
                        exit(1);
                    }
                }
                std::cout << "    PASS: Gradients for " << name << " are correct." << std::endl;
            };

            check_layer(block.conv1, "conv1");
            check_layer(block.conv2, "conv2");
            if (use_downsample) {
                // Accessing private downsample pointer
                check_layer(*(block.downsample), "downsample conv");
            }
        }
    }
};

void test_shape_matching_channels() {
    std::cout << "--- Test 1: Shape Correctness (Matching Channels) ---" << std::endl;
    ResidualBlock block(16, 16, 3, 1, 0.2);
    Tensor input(16, 64);
    Tensor output = block.forward(input);
    assert(output.get_channels() == 16 && output.get_width() == 64);
    std::cout << "PASS: Output shape is correct for matching channels." << std::endl;
}

void test_shape_downsampling_channels() {
    std::cout << "\n--- Test 2: Shape Correctness (Downsampling Channels) ---" << std::endl;
    ResidualBlock block(8, 16, 3, 1, 0.2);
    Tensor input(8, 64);
    Tensor output = block.forward(input);
    assert(output.get_channels() == 16 && output.get_width() == 64);
    std::cout << "PASS: Output shape is correct when downsampling is required." << std::endl;
}

int main() {
    try {
        test_shape_matching_channels();
        test_shape_downsampling_channels();
        
        // Invoke the friend class to run the private tests
        ResidualBlockTester::run_numerical_check();
        
        std::cout << "\n[SUCCESS] All ResidualBlock tests passed with high precision." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILURE] A test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
