// src/DataGenerator.cpp
#include "DataGenerator.h"
#include <iostream>

DataGenerator::DataGenerator(unsigned int seed) : rng(seed), dis(0.0f, 1.0f)
{
}

Tensor DataGenerator::generateRandomInput()
{
    Tensor input;
    input.shape = {3, 28, 28}; 
    input.data.resize(input.totalSize());
    
    for (size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = dis(rng); 
    }
    
    return input;
}

Tensor DataGenerator::generateRandomTarget()
{
    std::uniform_int_distribution<int> classDis(0, 9);
    int classIndex = classDis(rng);
    return classToOneHot(classIndex, 10);
}

Tensor DataGenerator::classToOneHot(int classIndex, int numClasses)
{
    Tensor oneHot;
    oneHot.shape = {static_cast<size_t>(numClasses)};
    oneHot.data.resize(numClasses, 0.0f);
    
    if (classIndex >= 0 && classIndex < numClasses) {
        oneHot.data[classIndex] = 1.0f;
    }
    
    return oneHot;
}

std::vector<std::pair<Tensor, Tensor>> DataGenerator::generateTrainingBatch(size_t batchSize)
{
    std::vector<std::pair<Tensor, Tensor>> batch;
    batch.reserve(batchSize);
    
    for (size_t i = 0; i < batchSize; ++i) {
        Tensor input = generateRandomInput();
        Tensor target = generateRandomTarget();
        batch.emplace_back(std::move(input), std::move(target));
    }
    
    return batch;
}
