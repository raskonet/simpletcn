#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

Tensor load_timeseries(const std::string& filepath, int max_length) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::vector<double> values;
    double val;
    while (file >> val) {
        values.push_back(val);
        if (max_length > 0 && values.size() >= static_cast<size_t>(max_length)) {
            break;
        }
    }
    
    if (values.empty()) {
        throw std::runtime_error("File was empty or invalid format: " + filepath);
    }

    // Create Tensor: 1 Channel, Width = values.size()
    Tensor t(1, static_cast<int>(values.size()));
    double* data = t.get_data();
    
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }
    
    std::cout << "[INFO] Loaded " << values.size() << " data points from " << filepath << std::endl;
    return t;
}
