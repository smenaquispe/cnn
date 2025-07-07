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
    // Pad a single channel (2D) within a tensor
    Tensor padChannel(const Tensor &input, size_t channelIdx)
    {
        if (padding == 0)
            return input;

        auto shape = input.getShape();
        if (shape.size() != 3) {
            throw std::invalid_argument("Input tensor must be 3D (channels, height, width)");
        }

        size_t channels = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t newH = H + 2 * padding;
        size_t newW = W + 2 * padding;

        Tensor padded;
        padded.shape = {channels, newH, newW};
        padded.data.resize(padded.totalSize(), 0.0f);

        // Copy all channels, padding the specified channel
        for (size_t c = 0; c < channels; ++c) {
            for (size_t i = 0; i < H; ++i) {
                for (size_t j = 0; j < W; ++j) {
                    if (c == channelIdx) {
                        padded.at({c, i + padding, j + padding}) = input.at({c, i, j});
                    } else {
                        padded.at({c, i, j}) = input.at({c, i, j});
                    }
                }
            }
        }

        return padded;
    }
};

#endif // POOLING_LAYER_H
