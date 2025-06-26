// src/activation_layers/Sigmoid.cpp
#include "activation_layers/Sigmoid.h"

vector<vector<vector<float>>> Sigmoid::apply(const vector<vector<vector<float>>> &input)
{
    auto output = input;
    for (auto &channel : output)
    {
        for (auto &row : channel)
        {
            for (auto &value : row)
            {
                value = 1.0f / (1.0f + exp(-value));
            }
        }
    }
    return output;
}