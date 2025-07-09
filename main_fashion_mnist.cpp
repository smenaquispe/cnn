// main_fashion_mnist.cpp
#include <iostream>
#include <vector>
#include <iomanip>
#include "NeuralNetwork.h"
#include "DataGenerator.h"
#include "utils/printMatrix.h"

using namespace std;

int main()
{
    cout << "=== CNN Training con Fashion-MNIST ===" << endl;
    cout << "Arquitectura:" << endl;
    cout << "Entrada 28x28x3 -> conv 3x3x3 (p:1,s:1) filtros 16 -> relu" << endl;
    cout << "-> max pooling 2x2 (s:2) -> conv 14x14 (p:1,s:1) filtros 4 -> relu" << endl;
    cout << "-> max pooling 2x2 (s:2) -> flatten 196 -> dense(16) -> softmax(10)" << endl;
    cout << "Learning Rate: 0.002" << endl;
    cout << "Épocas: 20" << endl;
    cout << "=================================================" << endl;

    NeuralNetwork network(0.002f);
    
    network.addConvLayer(16, 3, 1, 1);
    
    network.addReLULayer();
    
    network.addMaxPoolingLayer(2, 2);
    
    network.addConvLayer(4, 3, 1, 1);
    
    network.addReLULayer();
    
    network.addMaxPoolingLayer(2, 2);
    
    network.addFlattenLayer();
    
    network.addDenseLayer(196, 16);
    
    network.addDenseLayer(16, 10);
    network.addSoftmaxLayer();
    
    network.printArchitecture();
    
    DataGenerator dataGen(42);
    
    cout << "\n=== Cargando datos de Fashion-MNIST ===" << endl;
    if (!dataGen.loadFashionMnistCSV("fashion-mnist_test.csv")) {
        cerr << "Error: No se pudieron cargar los datos de Fashion-MNIST" << endl;
        return -1;
    }
    
    cout << "Dataset cargado con " << dataGen.getDatasetSize() << " muestras" << endl;
    
    cout << "\n=== Iniciando entrenamiento (20 épocas) ===" << endl;
    int numEpochs = 20;
    int batchSize = 32;
    
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        cout << "\nEpoch " << epoch + 1 << "/" << numEpochs << endl;
        
        size_t datasetSize = dataGen.getDatasetSize();
        int numBatches = max(1, (int)(datasetSize / batchSize));
        
        float totalLoss = 0.0f;
        int totalCorrect = 0;
        int totalSamples = 0;
        
        for (int batch = 0; batch < numBatches; ++batch) {
            auto batchData = dataGen.getFashionMnistBatch(batchSize);
            
            float batchLoss = 0.0f;
            int batchCorrect = 0;
            
            for (size_t i = 0; i < batchData.size(); ++i) {
                const auto &input = batchData[i].first;
                const auto &target = batchData[i].second;
                
                float loss = network.train(input, target);
                batchLoss += loss;
                
                Tensor prediction = network.predict(input);
                
                int predictedClass = 0;
                int actualClass = 0;
                float maxPred = prediction.data[0];
                float maxTarget = target.data[0];
                
                for (size_t j = 1; j < prediction.data.size(); ++j) {
                    if (prediction.data[j] > maxPred) {
                        maxPred = prediction.data[j];
                        predictedClass = j;
                    }
                    if (target.data[j] > maxTarget) {
                        maxTarget = target.data[j];
                        actualClass = j;
                    }
                }
                
                if (predictedClass == actualClass) {
                    batchCorrect++;
                }
                
                if (epoch == 0 && batch == 0 && i < 3) {
                    cout << "  Muestra " << i + 1 << ":" << endl;
                    cout << "    Clase real: " << actualClass << " (";
                    switch(actualClass) {
                        case 0: cout << "T-shirt/top"; break;
                        case 1: cout << "Trouser"; break;
                        case 2: cout << "Pullover"; break;
                        case 3: cout << "Dress"; break;
                        case 4: cout << "Coat"; break;
                        case 5: cout << "Sandal"; break;
                        case 6: cout << "Shirt"; break;
                        case 7: cout << "Sneaker"; break;
                        case 8: cout << "Bag"; break;
                        case 9: cout << "Ankle boot"; break;
                        default: cout << "Unknown"; break;
                    }
                    cout << ")" << endl;
                    cout << "    Clase predicha: " << predictedClass << endl;
                    cout << "    Loss: " << fixed << setprecision(4) << loss << endl;
                    cout << "    Confianza: " << fixed << setprecision(3) << maxPred << endl;
                }
            }
            
            totalLoss += batchLoss;
            totalCorrect += batchCorrect;
            totalSamples += batchData.size();
            
            if (batch % 5 == 0) {
                float batchAvgLoss = batchLoss / batchData.size();
                float batchAccuracy = (float)batchCorrect / batchData.size() * 100.0f;
                cout << "  Batch " << batch + 1 << "/" << numBatches 
                     << " - Loss: " << fixed << setprecision(4) << batchAvgLoss
                     << " - Acc: " << fixed << setprecision(1) << batchAccuracy << "%" << endl;
            }
        }
        
        float avgLoss = totalLoss / totalSamples;
        float accuracy = (float)totalCorrect / totalSamples * 100.0f;
        
        cout << "  === RESUMEN ÉPOCA " << epoch + 1 << " ===" << endl;
        cout << "  Loss promedio: " << fixed << setprecision(4) << avgLoss << endl;
        cout << "  Accuracy: " << fixed << setprecision(2) << accuracy << "%" << endl;
        cout << "  Muestras procesadas: " << totalSamples << endl;
    }
    
    cout << "\n=== Entrenamiento completado ===" << endl;
    
    cout << "\n=== Evaluación final ===" << endl;
    auto testBatch = dataGen.getFashionMnistBatch(10);
    
    int finalCorrect = 0;
    for (size_t i = 0; i < testBatch.size(); ++i) {
        const auto &input = testBatch[i].first;
        const auto &target = testBatch[i].second;
        
        Tensor prediction = network.predict(input);
        
        int predictedClass = 0;
        int actualClass = 0;
        float maxPred = prediction.data[0];
        float maxTarget = target.data[0];
        
        for (size_t j = 1; j < prediction.data.size(); ++j) {
            if (prediction.data[j] > maxPred) {
                maxPred = prediction.data[j];
                predictedClass = j;
            }
            if (target.data[j] > maxTarget) {
                maxTarget = target.data[j];
                actualClass = j;
            }
        }
        
        if (predictedClass == actualClass) {
            finalCorrect++;
        }
        
        cout << "Ejemplo " << i + 1 << ": ";
        cout << "Real=" << actualClass << ", Predicho=" << predictedClass;
        cout << ", Confianza=" << fixed << setprecision(3) << maxPred;
        cout << " (" << (predictedClass == actualClass ? "✓" : "✗") << ")" << endl;
    }
    
    float finalAccuracy = (float)finalCorrect / testBatch.size() * 100.0f;
    cout << "\nAccuracy final en muestra de prueba: " << fixed << setprecision(2) << finalAccuracy << "%" << endl;
    
    return 0;
}
