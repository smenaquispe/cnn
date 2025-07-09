// src/DenseLayer.cpp
#include "DenseLayer.h"
#include <cmath>
#include <algorithm>
#include <random>

DenseLayer::DenseLayer(size_t inputSize, size_t outputSize, float learningRate)
    : inputSize(inputSize), outputSize(outputSize), learningRate(learningRate)
{
    initWeights();
}

void DenseLayer::initWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = sqrt(6.0f / (inputSize + outputSize));
    std::uniform_real_distribution<float> dis(-limit, limit);
    
    weights.shape = {inputSize, outputSize};
    weights.data.resize(weights.totalSize());
    
    for (size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] = dis(gen);
    }
    
    biases.shape = {outputSize};
    biases.data.resize(outputSize, 0.0f);
}

Tensor DenseLayer::apply(const Tensor &input)
{
    auto inputShape = input.getShape();
    if (inputShape.size() != 1 || inputShape[0] != inputSize) {
        throw std::invalid_argument("Input shape must be [" + std::to_string(inputSize) + "]");
    }
    
    lastInput = input;
    
    Tensor output;
    output.shape = {outputSize};
    output.data.resize(outputSize, 0.0f);
    
    for (size_t j = 0; j < outputSize; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < inputSize; ++i) {
            sum += input.at({i}) * weights.at({i, j});
        }
        output.at({j}) = sum + biases.at({j});
    }
    
    lastOutput = output;
    return output;
}

void DenseLayer::backward(const Tensor &gradOutput)
{
    auto gradShape = gradOutput.getShape();
    
    if (gradShape.size() != 1) {
        throw std::invalid_argument("Gradient must be 1D");
    }
    
    if (gradShape[0] != outputSize) {
        return;
    }
    
    for (size_t i = 0; i < inputSize; ++i) {
        for (size_t j = 0; j < outputSize; ++j) {
            float grad = lastInput.at({i}) * gradOutput.at({j});
            weights.at({i, j}) -= learningRate * grad;
        }
    }
    
    for (size_t j = 0; j < outputSize; ++j) {
        biases.at({j}) -= learningRate * gradOutput.at({j});
    }
}
