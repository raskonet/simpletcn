#include <iostream>
#include <cstring>
#include <chrono>
#include <fstream> // FIX APPLIED: Added missing header
#include <cmath>
#include <algorithm> // for std::max
#include "tcn.hpp"
#include "utils.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

bool has_flag(int argc, char** argv, const char* flag) { for(int i=1; i<argc; ++i) if(strcmp(argv[i], flag)==0) return true; return false; }

void save_csv(const Tensor& y, const Tensor& p, const char* f) {
    std::ofstream out(f); 
    out << "Actual,Predicted\n";
    const double* a = y.get_data(); 
    const double* b = p.get_data();
    for(int i=std::max(0, y.get_width()-2000); i<y.get_width(); ++i) out << a[i] << "," << b[i] << "\n";
    std::cout << "Saved " << f << std::endl;
}

int main(int argc, char** argv) {
    bool use_gpu = has_flag(argc, argv, "--gpu");
    #ifndef USE_CUDA
    if(use_gpu) { std::cout << "⚠️ CUDA not compiled. Using CPU.\n"; use_gpu=false; }
    #endif

    std::cout << "=== TCN Hybrid Engine [" << (use_gpu?"GPU":"CPU") << "] ===\n";
    
    // Load Data
    Tensor raw = load_timeseries("ecg_train.txt", 2000); // Truncate for laptop speed
    int len = raw.get_width()-1;
    Tensor x(1, len); Tensor y(1, len);
    for(int i=0; i<len; ++i) { x.get_data()[i] = raw.get_data()[i]; y.get_data()[i] = raw.get_data()[i+1]; }
    
    if(use_gpu) x.to_device();

    // Small TCN for demonstration
    TCN model(1, 16, 3, 4, 1, 0.0);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<50; ++i) {
        model.zero_grad();
        const Tensor& pred = model.forward(x);
        
        // CPU Loss
        Tensor p_cpu = pred.clone();
        double loss=0; const double* p=p_cpu.get_data(); const double* t=y.get_data();
        for(int k=0; k<len; ++k) loss += (p[k]-t[k])*(p[k]-t[k]);
        
        if(i%10==0) std::cout << "Ep " << i << " MSE: " << loss/len << std::endl;
        
        // Backward
        Tensor grad(1, len); double* g=grad.get_data();
        for(int k=0; k<len; ++k) g[k] = 2.0*(p[k]-t[k])/len;
        model.backward(grad);
        model.update(0.001);
    }
    #ifdef USE_CUDA
    if(use_gpu) cudaDeviceSynchronize();
    #endif
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration<double>(end-start).count() << "s\n";
    
    const Tensor& final_p = model.forward(x);
    save_csv(y, final_p.clone(), "final_test_results.csv");
    return 0;
}
