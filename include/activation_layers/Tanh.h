// include/Tanh.h
#ifndef TANH_H
#define TANH_H

#include "ActivationLayer.h"
#include <cmath>

class Tanh : public ActivationLayer
{
public:
    ~Tanh() override = default;
    vector<vector<vector<float>>> apply(const vector<vector<vector<float>>> &input) override;
};

#endif // TANH_H
