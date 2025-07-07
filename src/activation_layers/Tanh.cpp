// src/activation_layers/Tanh.cpp
#include "activation_layers/Tanh.h"

Tensor Tanh::apply(const Tensor &input)
{
    Tensor output = input;
    for (auto &x : output.data)
    {
        x = std::tanh(x);
    }
    return output;
}