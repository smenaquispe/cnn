// src/DataGenerator.cpp
#include "DataGenerator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

DataGenerator::DataGenerator(unsigned int seed) : rng(seed), dis(0.0f, 1.0f), dataLoaded(false)
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

bool DataGenerator::loadFashionMnistCSV(const std::string& csvPath)
{
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo: " << csvPath << std::endl;
        return false;
    }
    
    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Error: Archivo CSV vacío" << std::endl;
        return false;
    }
    
    fashionMnistData.clear();
    int lineCount = 0;
    
    std::cout << "Cargando datos de Fashion-MNIST..." << std::endl;
    
    while (std::getline(file, line) && lineCount < 1000) { 
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        
        if (row.size() != 785) { 
            std::cerr << "Error: Línea " << lineCount + 2 << " tiene " << row.size() << " columnas en lugar de 785" << std::endl;
            continue;
        }
        
        int label = std::stoi(row[0]);
        
        std::vector<float> pixels;
        pixels.reserve(784);
        for (size_t i = 1; i < row.size(); ++i) {
            float pixelValue = std::stof(row[i]) / 255.0f; 
            pixels.push_back(pixelValue);
        }
        
        Tensor input = convertToTensor3D(pixels);
        Tensor target = classToOneHot(label, 10);
        
        fashionMnistData.emplace_back(std::move(input), std::move(target));
        lineCount++;
        
        if (lineCount % 100 == 0) {
            std::cout << "Cargadas " << lineCount << " muestras..." << std::endl;
        }
    }
    
    file.close();
    dataLoaded = true;
    
    std::cout << "Datos cargados exitosamente: " << fashionMnistData.size() << " muestras" << std::endl;
    return true;
}

Tensor DataGenerator::convertToTensor3D(const std::vector<float>& pixels)
{
    if (pixels.size() != 784) {
        throw std::invalid_argument("El vector debe tener exactamente 784 pixels (28x28)");
    }
    
    Tensor tensor;
    tensor.shape = {3, 28, 28};
    tensor.data.resize(tensor.totalSize());
    
    for (size_t c = 0; c < 3; ++c) {
        for (size_t i = 0; i < 28; ++i) {
            for (size_t j = 0; j < 28; ++j) {
                size_t srcIndex = i * 28 + j;
                size_t dstIndex = c * 28 * 28 + i * 28 + j;
                tensor.data[dstIndex] = pixels[srcIndex];
            }
        }
    }
    
    return tensor;
}

std::vector<std::pair<Tensor, Tensor>> DataGenerator::getFashionMnistBatch(size_t batchSize)
{
    if (!dataLoaded || fashionMnistData.empty()) {
        std::cerr << "Error: No hay datos de Fashion-MNIST cargados" << std::endl;
        return {};
    }
    
    std::vector<std::pair<Tensor, Tensor>> batch;
    batch.reserve(batchSize);
    
    std::uniform_int_distribution<size_t> indexDis(0, fashionMnistData.size() - 1);
    
    for (size_t i = 0; i < batchSize; ++i) {
        size_t randomIndex = indexDis(rng);
        batch.push_back(fashionMnistData[randomIndex]);
    }
    
    return batch;
}

size_t DataGenerator::getDatasetSize() const
{
    return fashionMnistData.size();
}
