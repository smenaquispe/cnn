// include/pooling_layers/AveragePooling.h
#ifndef AVERAGE_POOLING_H
#define AVERAGE_POOLING_H

#include "PoolingLayer.h"

class AveragePooling : public PoolingLayer
{
public:
    ~AveragePooling() override = default;
    AveragePooling(int poolH, int poolW, int stride, int padding = 0)
        : PoolingLayer(poolH, poolW, stride, padding) {}

    Tensor apply(const Tensor &input) override;
};

#endif // AVERAGE_POOLING_H
