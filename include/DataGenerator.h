// include/DataGenerator.h
#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include "Tensor.h"
#include <vector>
#include <random>

class DataGenerator
{
private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> dis;

public:
    DataGenerator(unsigned int seed = 42);
    
    Tensor generateRandomInput();
    
    Tensor generateRandomTarget();
    
    std::vector<std::pair<Tensor, Tensor>> generateTrainingBatch(size_t batchSize);
    
    Tensor classToOneHot(int classIndex, int numClasses = 10);
};

#endif // DATA_GENERATOR_H
