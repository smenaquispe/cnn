// include/activation_layers/Softmax.h
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "ActivationLayer.h"
#include <cmath>

class Softmax : public ActivationLayer
{
private:
    Tensor lastOutput; 

public:
    ~Softmax() override = default;
    Tensor apply(const Tensor &input) override;
    Tensor backward(const Tensor &gradOutput);
    const Tensor& getLastOutput() const { return lastOutput; }
};

#endif // SOFTMAX_H
