// include/activation_layers/ReLU.h
#ifndef RELU_H
#define RELU_H

#include "ActivationLayer.h"
#include <algorithm>

class ReLU : public ActivationLayer
{
public:
   ~ReLU() override = default;
   Tensor apply(const Tensor &input) override;
};

#endif // RELU_H
