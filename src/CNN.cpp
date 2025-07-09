#include "CNN.h"
#include <iostream>

Tensor CNN::addPadding(const Tensor &input)
{
    auto shape = input.getShape();
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::invalid_argument("Input tensor must be 2D or 3D");
    }

    if (padding == 0) {
        return input;
    }

    Tensor paddedTensor;
    
    if (shape.size() == 2) {
        // 2D case: H x W
        size_t H = shape[0];
        size_t W = shape[1];
        size_t newH = H + 2 * padding;
        size_t newW = W + 2 * padding;
        
        paddedTensor.shape = {newH, newW};
        paddedTensor.data.resize(paddedTensor.totalSize(), 0.0f);
        
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {
                paddedTensor.at({i + padding, j + padding}) = input.at({i, j});
            }
        }
    } else {
        size_t C = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t newH = H + 2 * padding;
        size_t newW = W + 2 * padding;
        
        paddedTensor.shape = {C, newH, newW};
        paddedTensor.data.resize(paddedTensor.totalSize(), 0.0f);
        
        for (size_t c = 0; c < C; ++c) {
            for (size_t i = 0; i < H; ++i) {
                for (size_t j = 0; j < W; ++j) {
                    paddedTensor.at({c, i + padding, j + padding}) = input.at({c, i, j});
                }
            }
        }
    }
    
    return paddedTensor;
}

Tensor CNN::convolve(const Tensor &inputTensor)
{
    const std::vector<size_t> &shape = inputTensor.getShape();

    if (shape.size() != 2 && shape.size() != 3)
    {
        throw std::invalid_argument("Solo se soportan entradas 2D (HxW) o 3D (CxHxW)");
    }

    size_t channels = (shape.size() == 3) ? shape[0] : 1;
    size_t input_h = (shape.size() == 3) ? shape[1] : shape[0];
    size_t input_w = (shape.size() == 3) ? shape[2] : shape[1];

    Tensor processedInput;
    if (shape.size() == 2) {
        processedInput.shape = {1, input_h, input_w};
        processedInput.data = inputTensor.data;
    } else {
        processedInput = inputTensor;
    }

    Tensor paddedInput = addPadding(processedInput);
    auto paddedShape = paddedInput.getShape();
    size_t padded_h = paddedShape[1];
    size_t padded_w = paddedShape[2];

    size_t num_filters = filters.size();
    size_t fh = filters[0].getHeight();
    size_t fw = filters[0].getWidth();
    size_t out_h = (padded_h - fh) / stride + 1;
    size_t out_w = (padded_w - fw) / stride + 1;

    Tensor output;
    output.shape = {num_filters, out_h, out_w};
    output.data.resize(output.totalSize(), 0.0f);

    for (size_t f = 0; f < num_filters; ++f)
    {
        for (size_t i = 0; i < out_h; ++i)
        {
            for (size_t j = 0; j < out_w; ++j)
            {
                float sum = 0.0f;
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t fi = 0; fi < fh; ++fi)
                    {
                        for (size_t fj = 0; fj < fw; ++fj)
                        {
                            size_t x = i * stride + fi;
                            size_t y = j * stride + fj;
                            sum += paddedInput.at({c, x, y}) * filters[f].getWeights().at({fi, fj});
                        }
                    }
                }
                output.at({f, i, j}) = sum;
            }
        }
    }

    return output;
}

Tensor CNN::applyLayers(const Tensor &input)
{
    Tensor output = input;
    for (auto &layer : layers)
    {
        output = layer->apply(output);
    }
    return output;
}

Tensor CNN::applyNextLayer(const Tensor &input)
{
    if (layers.empty())
    {
        throw std::runtime_error("No layers to apply.");
    }

    if (idxLayer >= layers.size())
    {
        throw std::runtime_error("No more layers to apply.");
    }

    Layer *layer = layers[idxLayer++];
    if (!layer)
    {
        throw std::runtime_error("Layer is null.");
    }

    return layer->apply(input);
}
