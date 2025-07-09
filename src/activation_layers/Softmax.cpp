// src/activation_layers/Softmax.cpp
#include "activation_layers/Softmax.h"
#include <algorithm>
#include <numeric>

Tensor Softmax::apply(const Tensor &input)
{
    auto shape = input.getShape();
    if (shape.size() != 1) {
        throw std::invalid_argument("Softmax expects 1D input");
    }
    
    Tensor output;
    output.shape = shape;
    output.data.resize(output.totalSize());
    
    float maxVal = *std::max_element(input.data.begin(), input.data.end());
    
    float sum = 0.0f;
    for (size_t i = 0; i < input.data.size(); ++i) {
        output.data[i] = std::exp(input.data[i] - maxVal);
        sum += output.data[i];
    }
    
    for (size_t i = 0; i < output.data.size(); ++i) {
        output.data[i] /= sum;
    }
    
    lastOutput = output;
    return output;
}

Tensor Softmax::backward(const Tensor &gradOutput)
{
    auto shape = gradOutput.getShape();
    if (shape.size() != 1) {
        throw std::invalid_argument("Gradient must be 1D");
    }
    
    size_t size = shape[0];
    Tensor gradInput;
    gradInput.shape = shape;
    gradInput.data.resize(size);
    
    for (size_t i = 0; i < size; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < size; ++j) {
            float delta = (i == j) ? 1.0f : 0.0f;
            sum += lastOutput.at({i}) * (delta - lastOutput.at({j})) * gradOutput.at({j});
        }
        gradInput.at({i}) = sum;
    }
    
    return gradInput;
}
