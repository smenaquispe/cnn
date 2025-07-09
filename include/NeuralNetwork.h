// include/NeuralNetwork.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Tensor.h"
#include "Filter.h"
#include "Layer.h"
#include "DenseLayer.h"
#include "activation_layers/Relu.h"
#include "activation_layers/Softmax.h"
#include "pooling_layers/MaxPooling.h"
#include "FlattenLayer.h"
#include "Loss.h"
#include <vector>
#include <memory>

class ConvLayer : public Layer
{
private:
    std::vector<Filter> filters;
    int stride;
    int padding;
    Tensor lastInput;
    Tensor lastOutput;
    float learningRate;

public:
    ConvLayer(int numFilters, int filterSize, int stride = 1, int padding = 0, float learningRate = 0.001f);
    ~ConvLayer() override = default;

    Tensor apply(const Tensor &input) override;
    void backward(const Tensor &gradOutput);
    void initWeights();
    
    // getters
    const std::vector<Filter>& getFilters() const { return filters; }
    const Tensor& getLastInput() const { return lastInput; }
    const Tensor& getLastOutput() const { return lastOutput; }
    
    // setters
    void setLearningRate(float lr) { learningRate = lr; }

private:
    Tensor addPadding(const Tensor &input);
    Tensor convolve(const Tensor &input);
};

class NeuralNetwork
{
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> lossFunction;
    float learningRate;

public:
    NeuralNetwork(float learningRate = 0.001f);
    ~NeuralNetwork() = default;

    void addConvLayer(int numFilters, int filterSize, int stride = 1, int padding = 0);
    void addReLULayer();
    void addMaxPoolingLayer(int poolSize, int stride);
    void addFlattenLayer();
    void addDenseLayer(int inputSize, int outputSize);
    void addSoftmaxLayer();
    
    Tensor forward(const Tensor &input);
    void backward(const Tensor &gradOutput);
    float train(const Tensor &input, const Tensor &target);
    Tensor predict(const Tensor &input);
    
    void setLearningRate(float lr) { learningRate = lr; }
    void printArchitecture();
};

#endif // NEURAL_NETWORK_H
