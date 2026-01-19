#include <iostream>
#include <cstring>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include "tcn.hpp"
#include "utils.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

bool has_flag(int argc, char** argv, const char* flag) { 
    for(int i=1; i<argc; ++i) if(strcmp(argv[i], flag)==0) return true; 
    return false; 
}

void save_csv(const Tensor& y, const Tensor& p, const char* f) {
    std::ofstream out(f); 
    out << "Actual,Predicted\n";
    const double* a = y.get_data(); 
    const double* b = p.get_data();
    for(int i=std::max(0, y.get_width()-2000); i<y.get_width(); ++i) out << a[i] << "," << b[i] << "\n";
    std::cout << "[IO] Saved results to " << f << std::endl;
}

int main(int argc, char** argv) {
    bool use_gpu = has_flag(argc, argv, "--gpu");
    bool inference_mode = has_flag(argc, argv, "--inference");

    #ifndef USE_CUDA
    if(use_gpu) { std::cout << "⚠️ CUDA not compiled. Using CPU.\n"; use_gpu=false; }
    #endif

    std::cout << "=== PrismTCN Hybrid Engine [" << (use_gpu?"GPU":"CPU") << "] ===\n";
    if (inference_mode) std::cout << "--- MODE: INFERENCE BENCHMARK ---\n";
    else std::cout << "--- MODE: TRAINING LOOP ---\n";

    int seq_len = 2000;
    Tensor x(1, seq_len); 
    Tensor y(1, seq_len);
    
    try {
        Tensor raw = load_timeseries("ecg_train.txt", seq_len);
        int len = raw.get_width()-1;
        x.reallocate(1, len);
        y.reallocate(1, len);
        for(int i=0; i<len; ++i) { 
            x.get_data()[i] = raw.get_data()[i]; 
            y.get_data()[i] = raw.get_data()[i+1]; 
        }
        seq_len = len;
    } catch (...) {
        std::cout << "⚠️  ecg_train.txt not found. Using Random Data for Benchmark.\n";
        for(int i=0; i<seq_len; ++i) {
            x.get_data()[i] = (double)rand() / RAND_MAX;
            y.get_data()[i] = (double)rand() / RAND_MAX;
        }
    }

    if(use_gpu) x.to_device();

    // Inputs=1, Channels=16, Kernel=3, Levels=4, Output=1, Dropout=0.0
    TCN model(1, 16, 3, 4, 1, 0.0);

    if (inference_mode) {
        model.set_training_mode(false); 

        std::cout << "Warmup..." << std::endl;
        for(int i=0; i<10; ++i) model.forward(x);
        
        #ifdef USE_CUDA
        if(use_gpu) cudaDeviceSynchronize();
        #endif

        int iters = 1000;
        std::cout << "Running " << iters << " iterations..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        
        for(int i=0; i<iters; ++i) {
            const Tensor& pred = model.forward(x);
            if (pred.get_channels() == -1) printf("Impossible");
        }

        #ifdef USE_CUDA
        if(use_gpu) cudaDeviceSynchronize();
        #endif

        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end-start).count();
        double avg_ms = total_ms / iters;

        std::cout << "------------------------------------------\n";
        std::cout << "[BENCHMARK] Total Time:    " << total_ms << " ms\n";
        std::cout << "[BENCHMARK] Avg Inference: " << avg_ms << " ms\n";
        std::cout << "------------------------------------------\n";

        const Tensor& final_p = model.forward(x);
        save_csv(y, final_p.clone(), "final_test_results.csv");

    } else {
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<50; ++i) {
            model.zero_grad();
            const Tensor& pred = model.forward(x);
            
            Tensor p_cpu = pred.clone();
            double loss=0; const double* p=p_cpu.get_data(); const double* t=y.get_data();
            for(int k=0; k<seq_len; ++k) loss += (p[k]-t[k])*(p[k]-t[k]);
            
            if(i%10==0) std::cout << "Ep " << i << " MSE: " << loss/seq_len << std::endl;
            
            Tensor grad(1, seq_len); double* g=grad.get_data();
            for(int k=0; k<seq_len; ++k) g[k] = 2.0*(p[k]-t[k])/seq_len;
            model.backward(grad);
            model.update(0.001);
        }
        #ifdef USE_CUDA
        if(use_gpu) cudaDeviceSynchronize();
        #endif
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Training Time: " << std::chrono::duration<double>(end-start).count() << "s\n";
        
        const Tensor& final_p = model.forward(x);
        save_csv(y, final_p.clone(), "final_test_results.csv");
    }

    return 0;
}
