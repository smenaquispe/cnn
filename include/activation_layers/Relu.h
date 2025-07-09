// include/activation_layers/ReLU.h
#ifndef RELU_H
#define RELU_H

#include "ActivationLayer.h"
#include <algorithm>

class ReLU : public ActivationLayer
{
private:
    Tensor lastInput; 

public:
   ~ReLU() override = default;
   Tensor apply(const Tensor &input) override;
   Tensor backward(const Tensor &gradOutput);
   const Tensor& getLastInput() const { return lastInput; }
};

#endif // RELU_H
