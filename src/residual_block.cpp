#include "residual_block.hpp"
#include "cuda_utils.hpp"
#include <cstring>
ResidualBlock::ResidualBlock(int in_c, int n_fil, int k, int d, double drop)
    : conv1(in_c, n_fil, k, d), relu1(), dropout1(drop),
      conv2(n_fil, n_fil, k, d), relu2(), dropout2(drop), downsample(nullptr) {
    if (in_c != n_fil) downsample = std::make_unique<Conv1D>(in_c, n_fil, 1, 1);
}
const Tensor& ResidualBlock::forward_ref(const Tensor& input) {
    const Tensor* x = &input;
    x = &conv1.forward_ref(*x); x = &relu1.forward_ref(*x); x = &dropout1.forward_ref(*x);
    x = &conv2.forward_ref(*x); x = &relu2.forward_ref(*x); x = &dropout2.forward_ref(*x);
    const Tensor* res = &input; if(downsample) res = &downsample->forward_ref(input);
    output_buffer.reallocate(x->get_channels(), x->get_width());
#ifdef USE_CUDA
    if (x->get_device() == Device::GPU) {
        if (res->get_device() == Device::CPU) const_cast<Tensor*>(res)->to_device();
        launch_add(*x, *res, output_buffer); return output_buffer;
    }
#endif
    const double* m = x->get_data(); const double* r = res->get_data(); double* o = output_buffer.get_data();
    for(size_t i=0; i<output_buffer.get_total_size(); ++i) o[i] = m[i] + r[i];
    return output_buffer;
}
Tensor ResidualBlock::forward(const Tensor& input) { return forward_ref(input).clone(); }
const Tensor& ResidualBlock::backward(const Tensor& grad_out) {
    const Tensor* g = &dropout2.backward(grad_out);
    g = &relu2.backward(*g); g = &conv2.backward(*g);
    g = &dropout1.backward(*g); g = &relu1.backward(*g); 
    const Tensor& g_main = conv1.backward(*g);
    
    grad_input_buffer.copy_from(g_main);

    if (downsample) {
        const Tensor& g_res = downsample->backward(grad_out);
#ifdef USE_CUDA
        if (grad_input_buffer.get_device() == Device::GPU) {
            launch_add(grad_input_buffer, g_res, grad_input_buffer);
        } else 
#endif
        {
            grad_input_buffer.to_host(); const_cast<Tensor&>(g_res).to_host();
            double* gm = grad_input_buffer.get_data(); const double* gr = g_res.get_data();
            for (size_t i = 0; i < grad_input_buffer.get_total_size(); ++i) gm[i] += gr[i];
        }
    } else {
#ifdef USE_CUDA
        if (grad_input_buffer.get_device() == Device::GPU) {
            launch_add(grad_input_buffer, grad_out, grad_input_buffer);
        } else 
#endif
        {
            grad_input_buffer.to_host(); const_cast<Tensor&>(grad_out).to_host();
            double* gm = grad_input_buffer.get_data(); const double* go = grad_out.get_data();
            for (size_t i = 0; i < grad_input_buffer.get_total_size(); ++i) gm[i] += go[i];
        }
    }
    return grad_input_buffer;
}
void ResidualBlock::update(double lr) { conv1.update(lr); conv2.update(lr); if(downsample) downsample->update(lr); }
void ResidualBlock::zero_grad() { conv1.zero_grad(); conv2.zero_grad(); if(downsample) downsample->zero_grad(); }
void ResidualBlock::set_training_mode(bool t) { dropout1.set_training_mode(t); dropout2.set_training_mode(t); }
void ResidualBlock::save(std::ofstream& out) const { conv1.save(out); conv2.save(out); bool has=downsample!=nullptr; out.write((char*)&has,1); if(has) downsample->save(out); }
void ResidualBlock::load(std::ifstream& in) { conv1.load(in); conv2.load(in); bool has; in.read((char*)&has,1); if(has){ if(!downsample) downsample=std::make_unique<Conv1D>(conv1.in_channels,conv1.out_channels,1,1); downsample->load(in); } }
