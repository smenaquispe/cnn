// include/Loss.h
#ifndef LOSS_H
#define LOSS_H

#include "Tensor.h"
#include <vector>

class Loss
{
public:
    virtual ~Loss() = default;
    virtual float computeLoss(const Tensor &predictions, const Tensor &targets) = 0;
    virtual Tensor computeGradient(const Tensor &predictions, const Tensor &targets) = 0;
};

class CrossEntropyLoss : public Loss
{
public:
    float computeLoss(const Tensor &predictions, const Tensor &targets) override;
    Tensor computeGradient(const Tensor &predictions, const Tensor &targets) override;
};

#endif // LOSS_H
