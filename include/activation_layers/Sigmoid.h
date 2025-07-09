// include/activation_layers/Sigmoid.h
#ifndef SIGMOID_H
#define SIGMOID_H

#include "ActivationLayer.h"
#include <cmath>

class Sigmoid : public ActivationLayer
{
public:
    ~Sigmoid() override = default;
    Tensor apply(const Tensor &input) override;
};

#endif // SIGMOID_H
