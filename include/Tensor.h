#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>

class Tensor {
public:
    std::vector<float> data;
    std::vector<size_t> shape;

    size_t totalSize() const {
        return std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());
    }

    size_t flattenIndex(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size())
            throw std::invalid_argument("Número de índices no coincide con la cantidad de dimensiones.");
        size_t index = 0, stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape[i])
                throw std::out_of_range("Índice fuera de los límites.");
            index += indices[i] * stride;
            stride *= shape[i];
        }
        return index;
    }

    float at(const std::vector<size_t>& indices) const {
        return data[flattenIndex(indices)];
    }

    float& at(const std::vector<size_t>& indices) {
        return data[flattenIndex(indices)];
    }

    float& at2D(size_t i, size_t j) {
        return at({i, j});
    }

    const std::vector<size_t>& getShape() const {
        return shape;
    }

    void printShape() const {
        std::cout << "Shape: [ ";
        for (auto d : shape) std::cout << d << " ";
        std::cout << "]" << std::endl;
    }
};

#endif // TENSOR_H
