// main.cpp
#include <iostream>
#include <vector>
#include "Filter.h"
#include "CNN.h"
#include "utils/printMatrix.h"
#include "activation_layers/Relu.h"
#include "pooling_layers/MaxPooling.h"
#include "FlattenLayer.h"
#include "utils/loadInputFromFile.h"

using namespace std;

int main()
{
    vector<Filter> filters;
    int num_filters = 3;
    for (int i = 0; i < num_filters; ++i)
    {
        Filter filter;
        filter.setWidth(3).setHeight(3).initWeights();
        printTensor(filter.getWeights(), "Filter " + to_string(i + 1) + " weights");
        filters.push_back(filter);
    }

    auto input = loadInputFromFileAsTensor("/home/smenaq/UNSA/cnn/input.txt");

    CNN cnn;
    ReLU relu; 
    MaxPooling maxPooling(2, 2, 2); 
    FlattenLayer flatten; 

    cnn.setFilters(filters)
        .setStride(3)
        .setPadding(2)
        .addLayer(&relu)
        .addLayer(&maxPooling)
        .addLayer(&flatten)
        .setInput(input);


    auto output = cnn.convolve(cnn.getInput());

    cout << "Convolution output:\n";
    printTensor(cnn.getInput(), "Input");
    
    printTensor(output, "Convolution - Feature maps");

    output = cnn.applyNextLayer(output);
    printTensor(output, "ReLU activation - Feature maps");

    output = cnn.applyNextLayer(output);
    printTensor(output, "MaxPooling - Feature maps");

    output = cnn.applyNextLayer(output);
    printFlattenedTensor(output, "Flattened output");


    return 0;
}
