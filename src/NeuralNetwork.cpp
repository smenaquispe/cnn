// src/NeuralNetwork.cpp
#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>
#include <random>

ConvLayer::ConvLayer(int numFilters, int filterSize, int stride, int padding, float learningRate)
    : stride(stride), padding(padding), learningRate(learningRate)
{
    filters.resize(numFilters);
    for (auto &filter : filters) {
        filter.setWidth(filterSize).setHeight(filterSize);
    }
    initWeights();
}

void ConvLayer::initWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 0.1f);
    
    for (auto &filter : filters) {
        filter.initWeights();
        // Add some random variation
        auto &weights = const_cast<Tensor&>(filter.getWeights());
        for (auto &weight : weights.data) {
            weight += dis(gen);
        }
    }
}

Tensor ConvLayer::addPadding(const Tensor &input)
{
    auto shape = input.getShape();
    if (shape.size() != 3) {
        throw std::invalid_argument("Input tensor must be 3D (channels, height, width)");
    }

    if (padding == 0) {
        return input;
    }

    size_t C = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    size_t newH = H + 2 * padding;
    size_t newW = W + 2 * padding;

    Tensor paddedTensor;
    paddedTensor.shape = {C, newH, newW};
    paddedTensor.data.resize(paddedTensor.totalSize(), 0.0f);

    for (size_t c = 0; c < C; ++c) {
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {
                paddedTensor.at({c, i + padding, j + padding}) = input.at({c, i, j});
            }
        }
    }

    return paddedTensor;
}

Tensor ConvLayer::convolve(const Tensor &input)
{
    auto shape = input.getShape();
    if (shape.size() != 3) {
        throw std::invalid_argument("Input tensor must be 3D (channels, height, width)");
    }

    size_t inputChannels = shape[0];
    size_t inputH = shape[1];
    size_t inputW = shape[2];

    // Apply padding
    Tensor paddedInput = addPadding(input);
    auto paddedShape = paddedInput.getShape();
    size_t paddedH = paddedShape[1];
    size_t paddedW = paddedShape[2];

    size_t numFilters = filters.size();
    size_t filterH = filters[0].getHeight();
    size_t filterW = filters[0].getWidth();
    size_t outputH = (paddedH - filterH) / stride + 1;
    size_t outputW = (paddedW - filterW) / stride + 1;

    Tensor output;
    output.shape = {numFilters, outputH, outputW};
    output.data.resize(output.totalSize(), 0.0f);

    for (size_t f = 0; f < numFilters; ++f) {
        for (size_t oh = 0; oh < outputH; ++oh) {
            for (size_t ow = 0; ow < outputW; ++ow) {
                float sum = 0.0f;
                for (size_t c = 0; c < inputChannels; ++c) {
                    for (size_t fh = 0; fh < filterH; ++fh) {
                        for (size_t fw = 0; fw < filterW; ++fw) {
                            size_t ih = oh * stride + fh;
                            size_t iw = ow * stride + fw;
                            if (ih < paddedH && iw < paddedW) {
                                sum += paddedInput.at({c, ih, iw}) * filters[f].getWeights().at({fh, fw});
                            }
                        }
                    }
                }
                output.at({f, oh, ow}) = sum;
            }
        }
    }

    return output;
}

Tensor ConvLayer::apply(const Tensor &input)
{
    lastInput = input;
    lastOutput = convolve(input);
    return lastOutput;
}

void ConvLayer::backward(const Tensor &gradOutput)
{   
    auto gradShape = gradOutput.getShape();
    auto inputShape = lastInput.getShape();
    
    if (gradShape.size() != 3 || inputShape.size() != 3) {
        throw std::invalid_argument("Gradient and input must be 3D tensors");
    }
    
    // Simple weight update (this is a simplified version)
    for (size_t f = 0; f < filters.size(); ++f) {
        auto &weights = const_cast<Tensor&>(filters[f].getWeights());
        for (size_t i = 0; i < weights.data.size(); ++i) {
            weights.data[i] -= learningRate * 0.001f; // Very simplified update
        }
    }
}

NeuralNetwork::NeuralNetwork(float learningRate) : learningRate(learningRate)
{
    lossFunction = std::make_unique<CrossEntropyLoss>();
}

void NeuralNetwork::addConvLayer(int numFilters, int filterSize, int stride, int padding)
{
    layers.push_back(std::make_unique<ConvLayer>(numFilters, filterSize, stride, padding, learningRate));
}

void NeuralNetwork::addReLULayer()
{
    layers.push_back(std::make_unique<ReLU>());
}

void NeuralNetwork::addMaxPoolingLayer(int poolSize, int stride)
{
    layers.push_back(std::make_unique<MaxPooling>(poolSize, poolSize, stride));
}

void NeuralNetwork::addFlattenLayer()
{
    layers.push_back(std::make_unique<FlattenLayer>());
}

void NeuralNetwork::addDenseLayer(int inputSize, int outputSize)
{
    layers.push_back(std::make_unique<DenseLayer>(inputSize, outputSize, learningRate));
}

void NeuralNetwork::addSoftmaxLayer()
{
    layers.push_back(std::make_unique<Softmax>());
}

Tensor NeuralNetwork::forward(const Tensor &input)
{
    Tensor output = input;
    for (auto &layer : layers) {
        output = layer->apply(output);
    }
    return output;
}

void NeuralNetwork::backward(const Tensor &gradOutput)
{
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (auto denseLayer = dynamic_cast<DenseLayer*>(layers[i].get())) {
            try {
                denseLayer->backward(gradOutput);
                break;
            } catch (const std::exception& e) {
                continue;
            }
        }
    }
}

float NeuralNetwork::train(const Tensor &input, const Tensor &target)
{
    Tensor output = forward(input);
    
    float loss = lossFunction->computeLoss(output, target);
    
    Tensor gradOutput = lossFunction->computeGradient(output, target);
    backward(gradOutput);
    
    return loss;
}

Tensor NeuralNetwork::predict(const Tensor &input)
{
    return forward(input);
}

void NeuralNetwork::printArchitecture()
{
    std::cout << "Neural Network Architecture:" << std::endl;
    std::cout << "=============================" << std::endl;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i + 1 << ": ";
        
        if (dynamic_cast<ConvLayer*>(layers[i].get())) {
            std::cout << "Convolutional Layer" << std::endl;
        } else if (dynamic_cast<ReLU*>(layers[i].get())) {
            std::cout << "ReLU Activation" << std::endl;
        } else if (dynamic_cast<MaxPooling*>(layers[i].get())) {
            std::cout << "Max Pooling Layer" << std::endl;
        } else if (dynamic_cast<FlattenLayer*>(layers[i].get())) {
            std::cout << "Flatten Layer" << std::endl;
        } else if (dynamic_cast<DenseLayer*>(layers[i].get())) {
            std::cout << "Dense Layer" << std::endl;
        } else if (dynamic_cast<Softmax*>(layers[i].get())) {
            std::cout << "Softmax Layer" << std::endl;
        } else {
            std::cout << "Unknown Layer" << std::endl;
        }
    }
    
    std::cout << "=============================" << std::endl;
}
