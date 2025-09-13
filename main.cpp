#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "tcn.hpp"

//simplistic loss function placeholder will implement actual one lateer
double calculate_mse_loss(const Tensor& prediction, const Tensor& target) {
    if (prediction.get_total_size() != target.get_total_size()) {
        throw std::runtime_error("Tensor size mismatch in loss calculation.");
    }
    
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

// Gradient of MSE
Tensor calculate_mse_grad(const Tensor& prediction, const Tensor& target) {
    if (prediction.get_total_size() != target.get_total_size()) {
        throw std::runtime_error("Tensor size mismatch in loss grad calculation.");
    }
    
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


int main() {
    const int sequence_length = 64;
    const int input_channels = 1;
    // For simplicity, we assume output channels match hidden channels for now.
    const int hidden_channels = 16; 
    const int output_channels = hidden_channels;
    const int kernel_size = 3;
    const int levels = 4; // Number of residual blocks (results in dilation 1, 2, 4, 8)
    const double learning_rate = 0.001;
    const int epochs = 10;
    const double dropout_rate = 0.2;

    std::cout << "TCN Framework Execution" << std::endl;
    std::cout << "Sequence Length: " << sequence_length << ", Levels: " << levels << ", Kernel Size: " << kernel_size << std::endl;

    TCN model(input_channels, hidden_channels, kernel_size, levels, dropout_rate);
    
    Tensor input(input_channels, sequence_length);
    Tensor target(output_channels, sequence_length);

    for(int i = 0; i < sequence_length; ++i) {
        input.get_data()[i] = static_cast<double>(i) / sequence_length;
        for (int j=0; j < output_channels; ++j) {
            target.get_data()[j * sequence_length + i] = static_cast<double>(i+1) / sequence_length; 
        }
    }


    std::cout << "\nStarting Training..." << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Set training mode for dropout
        model.set_training_mode(true);

        // 1. Zero gradients before the backward pass
        model.zero_grad();

        // 2. Forward pass returns the prediction
        Tensor prediction = model.forward(input);

        // 3. Calculate loss
        double loss = calculate_mse_loss(prediction, target);

        // 4. Calculate gradient of the loss
        Tensor loss_gradient = calculate_mse_grad(prediction, target);
        
        // 5. Backward pass
        model.backward(loss_gradient);
        
        // 6. Update weights
        model.update(learning_rate);

        std::cout << "Epoch: " << epoch + 1 << ", Loss: " << loss << std::endl;
    }

    std::cout << "\nTraining Finished." << std::endl;
    
    // Example of inference mode
    model.set_training_mode(false);
    Tensor final_prediction = model.forward(input);
    std::cout << "Inference complete." << std::endl;


    return 0;
}
