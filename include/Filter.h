#ifndef FILTER_H
#define FILTER_H

#include "Tensor.h"

class Filter
{
private:
    Tensor weights;  // Ahora usamos Tensor
    size_t width;
    size_t height;
    
public:
    Filter() : width(0), height(0) {}

    // Setters
    Filter &setWeights(const Tensor &w) {
        weights = w;
        return *this;
    }

    Filter &setWidth(size_t w) {
        width = w;
        return *this;
    }

    Filter &setHeight(size_t h) {
        height = h;
        return *this;
    }

    Filter &initWeights(const std::vector<size_t> &shape);  // CxHxW o HxW
    Filter &initWeights();  // Use width and height

    // Getters
    const Tensor &getWeights() const { return weights; }
    const std::vector<size_t> &getShape() const { return weights.shape; }
    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }
};

#endif // FILTER_H
