// include/FlattenLayer.h
#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "Layer.h"

class FlattenLayer : public Layer
{
private:
    vector<float> flattenedOutput;

public:
    ~FlattenLayer() override = default;
    FlattenLayer() = default;

    vector<vector<vector<float>>> apply(const vector<vector<vector<float>>> &input) override;
    void apply2D(const vector<vector<float>> &input);
    const vector<float> &getFlattenedOutput() const;
};

#endif // FLATTEN_LAYER_H
