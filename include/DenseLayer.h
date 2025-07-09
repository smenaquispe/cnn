// include/DenseLayer.h
#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "Layer.h"
#include "Tensor.h"
#include <vector>
#include <cstdlib>
#include <ctime>

class DenseLayer : public Layer
{
private:
    Tensor weights; 
    Tensor biases;  
    Tensor lastInput; 
    Tensor lastOutput; 
    size_t inputSize;
    size_t outputSize;
    float learningRate;

public:
    DenseLayer(size_t inputSize, size_t outputSize, float learningRate = 0.001f);
    ~DenseLayer() override = default;

    Tensor apply(const Tensor &input) override;
    void backward(const Tensor &gradOutput);
    void initWeights();
    
    // getters
    const Tensor& getWeights() const { return weights; }
    const Tensor& getBiases() const { return biases; }
    const Tensor& getLastInput() const { return lastInput; }
    const Tensor& getLastOutput() const { return lastOutput; }
    
    // setters
    void setLearningRate(float lr) { learningRate = lr; }
};

#endif // DENSE_LAYER_H
