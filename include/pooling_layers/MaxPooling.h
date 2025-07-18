// include/pooling_layers/MaxPooling.h
#ifndef MAX_POOLING_H
#define MAX_POOLING_H

#include "PoolingLayer.h"
#include <algorithm>

class MaxPooling : public PoolingLayer
{
public:
    ~MaxPooling() override = default;
    MaxPooling(int poolH, int poolW, int stride, int padding = 0)
        : PoolingLayer(poolH, poolW, stride, padding) {}

    Tensor apply(const Tensor &input) override;
};

#endif // MAX_POOLING_H
