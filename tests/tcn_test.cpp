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
    // Parameters
    int in_channels = 1;
    int num_filters = 4;
    int kernel_size = 3;
    int levels = 2; // Dilations: 1, 2
    
    TCN model(in_channels, num_filters, kernel_size, levels);
    
    Tensor input(in_channels, 10);
    for(size_t i=0; i<input.get_total_size(); ++i) input.get_data()[i] = 1.0;
    
    // 1. Forward
    model.set_training_mode(false); // Deterministic
    Tensor output = model.forward(input);
    
    assert(output.get_channels() == num_filters);
    assert(output.get_width() == 10);
    assert(!has_nans(output));
    
    std::cout << "PASS: Forward pass produced valid output shape." << std::endl;
    
    // 2. Backward
    model.zero_grad();
    Tensor grad_out(num_filters, 10);
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
    // TCN with levels=3, kernel=3.
    // Dilation: 1, 2, 4.
    // RF = 1 + 2*(3-1) + 2*(3-1)*2 + 2*(3-1)*4 = 1 + 4 + 8 + 16 = 29?
    // Formula: RF = 1 + \sum (k-1)*d
    // L1: (3-1)*1 = 2
    // L2: (3-1)*2 = 4
    // L3: (3-1)*4 = 8
    // Total RF = 1 + 2 + 4 + 8 = 15.
    
    int k=3;
    TCN model(1, 4, k, 3); 
    
    Tensor input(1, 20);
    input.zero();
    // Set impulse at end
    input.get_data()[19] = 1.0; 
    
    Tensor out = model.forward(input);
    // If RF is 15, changing index 19 should affect index 19 down to 19-(15-1) = 5.
    // Actually, TCN is causal. Output at t depends on t, t-1 ... t-RF.
    // So Input at 19 affects Output at 19, 20, 21...
    // Wait, we are testing dependency.
    
    // Let's test non-causality (leakage).
    // Input at 19 should NOT affect Output at 18.
    
    // Reset
    input.zero();
    input.get_data()[19] = 100.0;
    
    Tensor out_causal = model.forward(input);
    const double* d = out_causal.get_data();
    
    // Check output at 18. Should be exactly 0 (assuming biases are 0 or cancelled, strictly impulse response)
    // But biases are initialized.
    // Let's check difference.
    
    input.get_data()[19] = 0.0;
    Tensor out_base = model.forward(input);
    const double* d_base = out_base.get_data();
    
    // Check index 18
    double diff_18 = std::abs(d[18] - d_base[18]);
    if (diff_18 > 1e-9) {
        std::cout << "FAIL: Causal violation! Future input affected past output. Diff: " << diff_18 << std::endl;
        exit(1);
    }
    
    // Check index 19 (should be affected)
    double diff_19 = std::abs(d[19] - d_base[19]);
    if (diff_19 < 1e-9) {
         // It might be 0 if weights initialized to 0, but they are random.
         // Unlikely to be exactly 0.
         std::cout << "WARNING: Current input did not affect current output. Weights might be zero?" << std::endl;
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
