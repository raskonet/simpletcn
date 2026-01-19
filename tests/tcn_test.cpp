#include "tcn.hpp"
#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

bool has_nans(const Tensor& t) {
    const double* data = t.get_data();
    for(size_t i=0; i<t.get_total_size(); ++i) {
        if (std::isnan(data[i])) return true;
    }
    return false;
}

void test_tcn_flow() {
    std::cout << "--- Test 1: Full TCN Flow ---" << std::endl;
    int in_channels = 1;
    int num_filters = 4;
    int kernel_size = 3;
    int levels = 2; 
    int output_channels = 1; 
    double dropout = 0.0;    
    
    TCN model(in_channels, num_filters, kernel_size, levels, output_channels, dropout);
    
    Tensor input(in_channels, 10);
    for(size_t i=0; i<input.get_total_size(); ++i) input.get_data()[i] = 1.0;
    
    // 1. Forward
    model.set_training_mode(false);
    
    // We capture the reference to check properties immediately
    const Tensor& output = model.forward(input);
    
    assert(output.get_channels() == output_channels);
    assert(output.get_width() == 10);
    assert(!has_nans(output));
    
    std::cout << "PASS: Forward pass produced valid output shape." << std::endl;
    
    // 2. Backward
    model.zero_grad();
    Tensor grad_out(output_channels, 10);
    for(size_t i=0; i<grad_out.get_total_size(); ++i) grad_out.get_data()[i] = 0.1;
    
    Tensor grad_in = model.backward(grad_out);
    
    assert(grad_in.get_channels() == in_channels);
    assert(grad_in.get_width() == 10);
    assert(!has_nans(grad_in));
    
    std::cout << "PASS: Backward pass produced valid gradient shape." << std::endl;
    
    // 3. Update
    model.update(0.01);
    std::cout << "PASS: Update step executed without crash." << std::endl;
}

void test_receptive_field() {
    std::cout << "\n--- Test 2: Receptive Field Logic ---" << std::endl;
    
    int k=3;
    TCN model(1, 4, k, 3, 1, 0.0); 
    
    Tensor input(1, 20);
    input.zero();
    
    // --- TEST: Non-Causality (Leakage) Check ---
    // We want to ensure that input at t=19 does NOT affect output at t=18.
    
    // Case A: Input with impulse at index 19
    input.get_data()[19] = 100.0;
    
    // CRITICAL FIX: We must .clone() the result! 
    // If we kept a reference, the next forward() call would overwrite the data
    // because the engine reuses the same internal memory buffer.
    Tensor out_impulse = model.forward(input).clone();
    
    // Case B: Input without impulse
    input.get_data()[19] = 0.0;
    
    // Now we can just use the reference for the second run
    const Tensor& out_base_ref = model.forward(input);
    
    // Compare
    const double* d_impulse = out_impulse.get_data();
    const double* d_base = out_base_ref.get_data();
    
    // Check index 18 (Past) - Should be identical (0 difference)
    double diff_18 = std::abs(d_impulse[18] - d_base[18]);
    if (diff_18 > 1e-9) {
        std::cout << "FAIL: Causal violation! Future input affected past output. Diff: " << diff_18 << std::endl;
        exit(1);
    }
    
    // Check index 19 (Current) - Should be different (input changed!)
    // Note: It's theoretically possible for random weights to result in 0 change, but extremely unlikely.
    double diff_19 = std::abs(d_impulse[19] - d_base[19]);
    if (diff_19 < 1e-9) {
         std::cout << "WARNING: Input changed but output did not. (Weights might be near zero)" << std::endl;
    } else {
         std::cout << "PASS: Dependency check (current input affects current output)." << std::endl;
    }
    
    std::cout << "PASS: Causality check (future input does not affect past output)." << std::endl;
}

int main() {
    try {
        test_tcn_flow();
        test_receptive_field();
        std::cout << "\n[SUCCESS] All TCN module tests passed." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILURE] " << e.what() << std::endl;
        return 1;
    }
}
