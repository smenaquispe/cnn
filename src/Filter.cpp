// src/Filter.cpp
#include "Filter.h"
#include <cstdlib>
#include <vector>
#include <iostream>

using namespace std;

Filter &Filter::initWeights()
{
    weights.resize(height, vector<float>(width));
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            weights[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    return *this;
}
