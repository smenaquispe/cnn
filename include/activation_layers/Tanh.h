// include/activation_layers/Tanh.h
#ifndef TANH_H
#define TANH_H

#include "ActivationLayer.h"
#include <cmath>

class Tanh : public ActivationLayer
{
public:
    ~Tanh() override = default;
    Tensor apply(const Tensor &input) override;
};

#endif // TANH_H
