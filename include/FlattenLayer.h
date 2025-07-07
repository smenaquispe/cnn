// include/FlattenLayer.h
#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "Layer.h"
#include "Tensor.h"

class FlattenLayer : public Layer
{
private:
    Tensor flattenedOutput;

public:
    FlattenLayer() = default;
    ~FlattenLayer() override = default;

    Tensor apply(const Tensor &input) override;

    const Tensor &getFlattenedOutput() const;
};

#endif // FLATTEN_LAYER_H
