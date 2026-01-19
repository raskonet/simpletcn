#include "cuda_utils.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ double prism_atomic_add(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    // Pascal (GTX 10-series) and newer support native atomicAdd for double
    return atomicAdd(address, val);
#else
    // Manual CAS loop for older cards (Maxwell, Kepler)
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do { 
        assumed = old; 
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}


__global__ void conv1d_k(const double* __restrict__ in, const double* __restrict__ w, const double* __restrict__ b, double* __restrict__ out, int ic, int oc, int width, int ksize, int dil) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; 
    int o = blockIdx.y * blockDim.y + threadIdx.y; 
    if (t < width && o < oc) {
        double sum = b[o];
        int p = (ksize - 1) * dil; 
        for (int i = 0; i < ic; ++i) {
            for (int k = 0; k < ksize; ++k) {
                int in_t = t - (p - k * dil);
                if (in_t >= 0 && in_t < width) sum += in[i * width + in_t] * w[(o * ic + i) * ksize + k];
            }
        }
        out[o * width + t] = sum;
    }
}
void launch_conv1d(const Tensor& input, const Tensor& weights, const Tensor& bias, Tensor& output, int dilation, int kernel_size) {
    dim3 block(32, 4); dim3 grid((output.get_width()+31)/32, (output.get_channels()+3)/4);
    conv1d_k<<<grid, block>>>(input.get_device_data(), weights.get_device_data(), bias.get_device_data(), output.get_device_data(), input.get_channels(), output.get_channels(), output.get_width(), kernel_size, dilation);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void relu_k(double* d, int s) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<s && d[i]<0.0) d[i]=0.0; }
void launch_relu(Tensor& t) { relu_k<<<(t.get_total_size()+255)/256, 256>>>(t.get_device_data(), t.get_total_size()); CUDA_CHECK(cudaDeviceSynchronize()); }

__global__ void add_k(const double* a, const double* b, double* o, int s) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<s) o[i]=a[i]+b[i]; }
void launch_add(const Tensor& a, const Tensor& b, Tensor& o) { add_k<<<(a.get_total_size()+255)/256, 256>>>(a.get_device_data(), b.get_device_data(), o.get_device_data(), a.get_total_size()); CUDA_CHECK(cudaDeviceSynchronize()); }

__global__ void drop_k(const double* i, const double* m, double* o, int s, double sc) { int idx = blockIdx.x*blockDim.x+threadIdx.x; if(idx<s) o[idx]=i[idx]*m[idx]*sc; }
void launch_dropout_apply(const Tensor& i, const Tensor& m, Tensor& o, double sc) { drop_k<<<(i.get_total_size()+255)/256, 256>>>(i.get_device_data(), m.get_device_data(), o.get_device_data(), i.get_total_size(), sc); CUDA_CHECK(cudaDeviceSynchronize()); }

__global__ void conv1d_bw_k(const double* __restrict__ grad_out, const double* __restrict__ input, const double* __restrict__ weights, double* __restrict__ grad_in, double* __restrict__ grad_w, double* __restrict__ grad_b, int ic, int oc, int width, int ksize, int dil) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; 
    int o = blockIdx.y * blockDim.y + threadIdx.y; 
    if (t < width && o < oc) {
        double g = grad_out[o * width + t];
        
        // Use our wrapper
        prism_atomic_add(&grad_b[o], g);

        int p = (ksize - 1) * dil;
        for (int i = 0; i < ic; ++i) {
            for (int k = 0; k < ksize; ++k) {
                int in_t = t - (p - k * dil);
                if (in_t >= 0 && in_t < width) {
                    // Use our wrapper
                    prism_atomic_add(&grad_w[(o * ic + i) * ksize + k], g * input[i * width + in_t]);
                    prism_atomic_add(&grad_in[i * width + in_t], g * weights[(o * ic + i) * ksize + k]);
                }
            }
        }
    }
}

void launch_conv1d_backward(const Tensor& grad_out, const Tensor& input, const Tensor& weights, Tensor& grad_in, Tensor& grad_w, Tensor& grad_b, int dilation, int kernel_size) {
    cudaMemset(grad_in.get_device_data(), 0, grad_in.get_total_size() * sizeof(double));
    dim3 block(32, 4);
    dim3 grid((grad_out.get_width() + 31) / 32, (grad_out.get_channels() + 3) / 4);
    conv1d_bw_k<<<grid, block>>>(grad_out.get_device_data(), input.get_device_data(), weights.get_device_data(), grad_in.get_device_data(), grad_w.get_device_data(), grad_b.get_device_data(), input.get_channels(), grad_out.get_channels(), grad_out.get_width(), kernel_size, dilation);
    CUDA_CHECK(cudaDeviceSynchronize());
}
__global__ void relu_bw_k(const double* grad_out, const double* input, double* grad_in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) grad_in[i] = (input[i] > 0.0) ? grad_out[i] : 0.0;
}
void launch_relu_backward(const Tensor& grad_out, const Tensor& input_cache, Tensor& grad_in) {
    relu_bw_k<<<(grad_out.get_total_size() + 255)/256, 256>>>(grad_out.get_device_data(), input_cache.get_device_data(), grad_in.get_device_data(), grad_out.get_total_size());
    CUDA_CHECK(cudaDeviceSynchronize());
}
__global__ void sgd_k(double* params, const double* grads, int size, double lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) params[i] -= lr * grads[i];
}
void launch_sgd_update(Tensor& params, const Tensor& grads, double lr) {
    sgd_k<<<(params.get_total_size() + 255)/256, 256>>>(params.get_device_data(), grads.get_device_data(), params.get_total_size(), lr);
    CUDA_CHECK(cudaDeviceSynchronize());
}
#endif
