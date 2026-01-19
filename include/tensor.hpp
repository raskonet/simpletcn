#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <string>
#include <cstddef> 
#include <fstream>

enum class Device { CPU, GPU };

class Tensor {
private:
    double* data;         // Host Data
    double* device_data;  // Device Data
    int channels;
    int width;
    size_t total_size;
    Device current_device;

public:
    Tensor(int channels, int width);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor clone() const;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Accessors
    double* get_data();
    const double* get_data() const;
    double* get_device_data() const; 
    
    int get_channels() const { return channels; }
    int get_width() const { return width; }
    size_t get_total_size() const { return total_size; }
    Device get_device() const { return current_device; }

    // Memory Ops
    void to_device();
    void to_host();
    void zero(); 

    void save(std::ofstream& out) const;
    void load(std::ifstream& in);
};
#endif
