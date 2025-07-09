// src/FlattenLayer.cpp
#include "FlattenLayer.h"

Tensor FlattenLayer::apply(const Tensor &input)
{
    flattenedOutput.data = input.data;
    flattenedOutput.shape = {input.totalSize()}; 
    return flattenedOutput;
}

const Tensor &FlattenLayer::getFlattenedOutput() const
{
    return flattenedOutput;
}
