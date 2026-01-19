#include <iostream>
#include <cstring>
#include <chrono>
#include "tcn.hpp"
#include "utils.hpp"

bool has_flag(int argc, char** argv, const char* flag) {
    for(int i=1; i<argc; ++i) if(strcmp(argv[i], flag)==0) return true;
    return false;
}

int main(int argc, char** argv) {
    bool use_gpu = has_flag(argc, argv, "--gpu");
    std::cout << "=== TCN Training [" << (use_gpu ? "GPU/CUDA" : "CPU/OpenMP") << "] ===" << std::endl;

    // Load Data (Assumes 2000 points exist in ecg_train.txt)
    Tensor train_data(1, 1);
    try {
        train_data = load_timeseries("ecg_train.txt", 2000); 
    } catch (...) {
        std::cerr << "Error: ecg_train.txt not found. Run analysis/preprocess_ptbxl.py or fetch_data.py first." << std::endl;
        // Mock data if file missing
        train_data = Tensor(1, 100);
        train_data.zero();
    }
    
    int len = train_data.get_width() - 1;
    if(len < 10) len = 99; // Safety

    Tensor x(1, len);
    Tensor y(1, len);
    const double* raw = train_data.get_data();
    for(int i=0; i<len; ++i) {
        x.get_data()[i] = raw[i];
        y.get_data()[i] = raw[i+1];
    }

    if(use_gpu) {
        std::cout << "Moving training data to VRAM..." << std::endl;
        x.to_device();
    }

    TCN model(1, 16, 3, 4, 1, 0.0);

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<50; ++i) {
        model.zero_grad();
        Tensor pred = model.forward(x);
        
        // Pull pred back to CPU for Loss/Backprop (Hybrid Mode)
        const double* p = pred.get_data();
        const double* t = y.get_data();
        
        double loss = 0;
        for(int j=0; j<len; ++j) loss += (p[j]-t[j])*(p[j]-t[j]);
        loss /= len;
        
        if(i % 10 == 0) std::cout << "Iter " << i << " Loss: " << loss << std::endl;

        Tensor grad(1, len);
        double* g = grad.get_data();
        for(int j=0; j<len; ++j) g[j] = 2.0*(p[j]-t[j])/len;
        
        model.backward(grad);
        model.update(0.001);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << "s" << std::endl;

    return 0;
}
