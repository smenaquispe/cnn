// include/Layer.h
#ifndef LAYER_H
#define LAYER_H

#include "Tensor.h"

class Layer
{
public:
    virtual ~Layer() = default;

    virtual Tensor apply(const Tensor &input) = 0;
};

#endif // LAYER_H
