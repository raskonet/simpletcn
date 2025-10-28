#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <string>
#include <cstddef> 
#include <fstream>

class Tensor {
private:
    double* data;
    int channels;
    int width;
    size_t total_size;

public:
    Tensor(int channels, int width);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor clone() const;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    double* get_data();
    const double* get_data() const;
    int get_channels() const;
    int get_width() const;
    size_t get_total_size() const;
    std::string dimensions_str() const;
    
    void zero(); 

    void save(std::ofstream& out) const;
    void load(std::ifstream& in);
};

#endif
