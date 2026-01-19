#include "tcn.hpp"
#include <iostream>

TCN::TCN(int ic, int nc, int k, int l, int oc, double d) : final_conv(nc, oc, 1, 1) {
    int cur = ic; int dil = 1;
    for(int i=0; i<l; ++i) { blocks.emplace_back(cur, nc, k, dil, d); cur=nc; dil*=2; }
}

const Tensor& TCN::forward(const Tensor& input) {
    const Tensor* c = &input;
    for(auto& b : blocks) c = &b.forward_ref(*c);
    c = &final_conv.forward_ref(*c);
    return *c;
}

Tensor TCN::backward(const Tensor& g) { 
    const Tensor* x = &final_conv.backward(g); 
    for(int i=blocks.size()-1; i>=0; --i) x = &blocks[i].backward(*x); 
    return x->clone(); 
}

void TCN::update(double lr) { for(auto& b:blocks) b.update(lr); final_conv.update(lr); }
void TCN::zero_grad() { for(auto& b:blocks) b.zero_grad(); final_conv.zero_grad(); }
void TCN::set_training_mode(bool t) { for(auto& b:blocks) b.set_training_mode(t); }
void TCN::save(const std::string& f) const { std::ofstream o(f, std::ios::binary); for(const auto& b:blocks) b.save(o); final_conv.save(o); }

void TCN::load(const std::string& f) { 
    std::ifstream i(f, std::ios::binary); 
    for(auto& b:blocks) b.load(i); 
    final_conv.load(i); 
}
