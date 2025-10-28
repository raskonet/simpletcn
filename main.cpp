#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include "tensor.hpp"
#include "tcn.hpp"
#include "utils.hpp"

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
    
    double scale = 2.0 / n;
    for (size_t i = 0; i < n; ++i) {
        grad_data[i] = scale * (pred_data[i] - target_data[i]); 
    }
    return grad;
}

double evaluate(TCN& model, const Tensor& input, const Tensor& target) {
    model.set_training_mode(false);
    Tensor pred = model.forward(input);
    return calculate_mse_loss(pred, target);
}

void save_predictions_csv(const Tensor& actual, const Tensor& predicted, const std::string& filename) {
    std::ofstream file(filename);
    file << "Index,Actual,Predicted\n";
    const double* act = actual.get_data();
    const double* pred = predicted.get_data();
    size_t n = actual.get_total_size();
    size_t start = (n > 2000) ? n - 2000 : 0;
    
    for (size_t i = start; i < n; ++i) {
        file << i << "," << act[i] << "," << pred[i] << "\n";
    }
    std::cout << "[IO] Predictions saved to " << filename << std::endl;
}

int main() {
    std::cout << "=== TCN Production Training ===" << std::endl;

    Tensor train_data = load_timeseries("ecg_train.txt", -1);
    Tensor val_data   = load_timeseries("ecg_val.txt", -1);
    Tensor test_data  = load_timeseries("ecg_test.txt", -1);

    auto prepare_xy = [](const Tensor& raw) -> std::pair<Tensor, Tensor> {
        int len = raw.get_width() - 1;
        Tensor x(1, len);
        Tensor y(1, len);
        const double* r = raw.get_data();
        for(int i=0; i<len; ++i) {
            x.get_data()[i] = r[i];
            y.get_data()[i] = r[i+1];
        }
        return {std::move(x), std::move(y)};
    };

    auto [train_in, train_tgt] = prepare_xy(train_data);
    auto [val_in, val_tgt]     = prepare_xy(val_data);
    auto [test_in, test_tgt]   = prepare_xy(test_data);

    TCN model(1, 24, 7, 5, 1, 0.15); 

    const int epochs = 1000;
    const double clip = 5.0;
    double learning_rate = 0.002;
    double best_val_loss = std::numeric_limits<double>::infinity();
    std::string model_path = "best_tcn_model.bin";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (epoch == 200) learning_rate = 0.0005;
        if (epoch == 600) learning_rate = 0.0001;
        if (epoch == 900) learning_rate = 0.00002;

        model.set_training_mode(true);
        model.zero_grad();
        Tensor train_pred = model.forward(train_in);
        double train_loss = calculate_mse_loss(train_pred, train_tgt);
        Tensor loss_grad = calculate_mse_grad(train_pred, train_tgt);
        model.backward(loss_grad);
        model.clip_gradients(clip);
        model.update(learning_rate);

        double val_loss = evaluate(model, val_in, val_tgt);

        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            model.save(model_path);
        }

        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << std::setw(4) << epoch + 1 
                      << " | Train: " << std::fixed << std::setprecision(5) << train_loss
                      << " | Val: " << val_loss
                      << " | Best: " << best_val_loss 
                      << " | LR: " << std::defaultfloat << learning_rate << std::endl;
        }
    }

    std::cout << "\n[Train] Finished. Best Val Loss: " << best_val_loss << std::endl;
    model.load(model_path);
    
    model.set_training_mode(false);
    Tensor test_pred = model.forward(test_in);
    double final_test_loss = calculate_mse_loss(test_pred, test_tgt);
    
    std::cout << "FINAL TEST SET MSE: " << final_test_loss << std::endl;
    save_predictions_csv(test_tgt, test_pred, "final_test_results.csv");

    return 0;
}
