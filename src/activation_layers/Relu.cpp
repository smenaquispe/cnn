// src/activation_layers/Relu.cpp
#include "activation_layers/Relu.h"


Tensor ReLU::apply(const Tensor &input)
{
    Tensor output = input;
    for (auto &val : output.data)
    {
        val = std::max(0.0f, val);
    }
    return output;
}