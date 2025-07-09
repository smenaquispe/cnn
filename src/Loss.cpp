// src/Loss.cpp
#include "Loss.h"
#include <cmath>
#include <algorithm>

float CrossEntropyLoss::computeLoss(const Tensor &predictions, const Tensor &targets)
{
    auto predShape = predictions.getShape();
    auto targetShape = targets.getShape();
    
    if (predShape.size() != 1 || targetShape.size() != 1 || predShape[0] != targetShape[0]) {
        throw std::invalid_argument("Predictions and targets must have same 1D shape");
    }
    
    float loss = 0.0f;
    for (size_t i = 0; i < predShape[0]; ++i) {
        float pred = std::max(predictions.at({i}), 1e-15f); // Avoid log(0)
        loss -= targets.at({i}) * std::log(pred);
    }
    
    return loss;
}

Tensor CrossEntropyLoss::computeGradient(const Tensor &predictions, const Tensor &targets)
{
    auto predShape = predictions.getShape();
    auto targetShape = targets.getShape();
    
    if (predShape.size() != 1 || targetShape.size() != 1 || predShape[0] != targetShape[0]) {
        throw std::invalid_argument("Predictions and targets must have same 1D shape");
    }
    
    Tensor gradient;
    gradient.shape = predShape;
    gradient.data.resize(gradient.totalSize());
    
    for (size_t i = 0; i < predShape[0]; ++i) {
        float pred = std::max(predictions.at({i}), 1e-15f); // Avoid division by 0
        gradient.at({i}) = -targets.at({i}) / pred;
    }
    
    return gradient;
}
