// src/FlattenLayer.cpp
#include "FlattenLayer.h"

Tensor FlattenLayer::apply(const Tensor &input)
{
    flattenedOutput.data = input.data;
    flattenedOutput.shape = {input.totalSize()}; // 1D: vector con todos los valores
    return flattenedOutput;
}

const Tensor &FlattenLayer::getFlattenedOutput() const
{
    return flattenedOutput;
}
