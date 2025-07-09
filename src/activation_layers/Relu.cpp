// src/activation_layers/Relu.cpp
#include "activation_layers/Relu.h"

Tensor ReLU::apply(const Tensor &input)
{
    lastInput = input; 
    
    Tensor output = input;
    for (auto &val : output.data)
    {
        val = std::max(0.0f, val);
    }
    return output;
}

Tensor ReLU::backward(const Tensor &gradOutput)
{
    auto shape = gradOutput.getShape();
    Tensor gradInput;
    gradInput.shape = shape;
    gradInput.data.resize(gradInput.totalSize());
    
    for (size_t i = 0; i < gradInput.data.size(); ++i) {
        gradInput.data[i] = (lastInput.data[i] > 0.0f) ? gradOutput.data[i] : 0.0f;
    }
    
    return gradInput;
}