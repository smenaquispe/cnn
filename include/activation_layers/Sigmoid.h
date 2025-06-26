// include/Sigmoid.h
#ifndef SIGMOID_H
#define SIGMOID_H

#include "ActivationLayer.h"
#include <cmath>

class Sigmoid : public ActivationLayer
{
public:
    ~Sigmoid() override = default;
    vector<vector<vector<float>>> apply(const vector<vector<vector<float>>> &input) override;
};

#endif // SIGMOID_H
