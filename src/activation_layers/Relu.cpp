// src/activation_layers/Relu.cpp
#include "activation_layers/Relu.h"


vector<vector<vector<float>>> ReLU::apply(const vector<vector<vector<float>>> &input)
{
    auto output = input;
    for (auto &channel : output)
    {
        for (auto &row : channel)
        {
            for (auto &value : row)
            {
                value = std::max(0.0f, value);
            }
        }
    }
    return output;
}