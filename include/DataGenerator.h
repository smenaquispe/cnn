// include/DataGenerator.h
#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include "Tensor.h"
#include <vector>
#include <random>
#include <string>

class DataGenerator
{
private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> dis;
    std::vector<std::pair<Tensor, Tensor>> fashionMnistData;
    bool dataLoaded;

public:
    DataGenerator(unsigned int seed = 42);
    
    Tensor generateRandomInput();
    
    Tensor generateRandomTarget();
    
    std::vector<std::pair<Tensor, Tensor>> generateTrainingBatch(size_t batchSize);
    
    Tensor classToOneHot(int classIndex, int numClasses = 10);
    
    bool loadFashionMnistCSV(const std::string& csvPath);
    
    Tensor convertToTensor3D(const std::vector<float>& pixels);
    
    std::vector<std::pair<Tensor, Tensor>> getFashionMnistBatch(size_t batchSize);
    
    size_t getDatasetSize() const;
};

#endif // DATA_GENERATOR_H
