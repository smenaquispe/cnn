// src/pooling_layers/AveragePooling.cpp
#include "pooling_layers/AveragePooling.h"

Tensor AveragePooling::apply(const Tensor &input)
{
    auto shape = input.getShape();
    if (shape.size() != 3) {
        throw std::invalid_argument("Input tensor must be 3D (channels, height, width)");
    }

    size_t channels = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];

    Tensor processedInput = input;
    if (padding > 0) {
        size_t newH = H + 2 * padding;
        size_t newW = W + 2 * padding;
        
        processedInput.shape = {channels, newH, newW};
        processedInput.data.resize(processedInput.totalSize(), 0.0f);
        
        // Copy with padding
        for (size_t c = 0; c < channels; ++c) {
            for (size_t i = 0; i < H; ++i) {
                for (size_t j = 0; j < W; ++j) {
                    processedInput.at({c, i + padding, j + padding}) = input.at({c, i, j});
                }
            }
        }
        
        H = newH;
        W = newW;
    }

    size_t outH = (H - poolHeight) / stride + 1;
    size_t outW = (W - poolWidth) / stride + 1;

    Tensor output;
    output.shape = {channels, outH, outW};
    output.data.resize(output.totalSize(), 0.0f);

    for (size_t c = 0; c < channels; ++c) {
        for (size_t i = 0; i < outH; ++i) {
            for (size_t j = 0; j < outW; ++j) {
                float sum = 0.0f;
                int count = 0;
                for (int m = 0; m < poolHeight; ++m) {
                    for (int n = 0; n < poolWidth; ++n) {
                        size_t y = i * stride + m;
                        size_t x = j * stride + n;
                        if (y < H && x < W) {
                            sum += processedInput.at({c, y, x});
                            count++;
                        }
                    }
                }
                output.at({c, i, j}) = sum / count;
            }
        }
    }

    return output;
}