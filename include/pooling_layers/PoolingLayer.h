// include/pooling_layers/PoolingLayer.h
#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <vector>
#include "Layer.h"
using namespace std;

class PoolingLayer : public Layer
{
protected:
    int poolHeight;
    int poolWidth;
    int stride;
    int padding;

public:
    PoolingLayer(int poolH, int poolW, int stride, int padding = 0)
        : poolHeight(poolH), poolWidth(poolW), stride(stride), padding(padding) {}

    virtual ~PoolingLayer() = default;

protected:
    vector<vector<float>> pad(const vector<vector<float>> &input)
    {
        if (padding == 0)
            return input;

        int H = input.size();
        int W = input[0].size();
        int newH = H + 2 * padding;
        int newW = W + 2 * padding;

        vector<vector<float>> padded(newH, vector<float>(newW, 0.0f));

        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                padded[i + padding][j + padding] = input[i][j];

        return padded;
    }
};

#endif // POOLING_LAYER_H
