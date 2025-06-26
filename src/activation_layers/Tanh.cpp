// src/activation_layers/Tanh.cpp
#include "activation_layers/Tanh.h"

vector<vector<vector<float>>> Tanh::apply(const vector<vector<vector<float>>> &input)
{
    auto output = input;
    for (auto &channel : output)
    {
        for (auto &row : channel)
        {
            for (auto &value : row)
            {
                value = tanh(value);
            }
        }
    }
    return output;
}