#ifndef TCN_HPP
#define TCN_HPP
#include "residual_block.hpp"
#include <vector>
class TCN {
    std::vector<ResidualBlock> blocks; Conv1D final_conv;
public:
    TCN(int ic, int nc, int k, int l, int oc, double d);
    const Tensor& forward(const Tensor& input);
    Tensor backward(const Tensor& g);
    void update(double lr); void zero_grad(); void set_training_mode(bool t);
    void save(const std::string& f) const; void load(const std::string& f);
};
#endif
