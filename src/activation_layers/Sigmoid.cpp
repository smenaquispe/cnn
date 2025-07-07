// src/activation_layers/Sigmoid.cpp
#include "activation_layers/Sigmoid.h"

Tensor Sigmoid::apply(const Tensor &input)
{
    Tensor output = input;
    for (auto &x : output.data)
    {
        x = 1.0f / (1.0f + std::exp(-x));
    }
    return output;
}