#include <iostream>
#include <vector>
#include "Filter.h"
#include "CNN.h"
#include "utils/printMatrix.h"
#include "activation_layers/Relu.h"
#include "pooling_layers/MaxPooling.h"
#include "FlattenLayer.h"

using namespace std;

int main()
{
    vector<Filter> filters;
    int num_filters = 3;
    for (int i = 0; i < num_filters; ++i)
    {
        Filter filter;
        filter.setWidth(3).setHeight(3).initWeights();
        printMatrix(filter.getWeights(), "Filter " + to_string(i + 1) + " weights");
        filters.push_back(filter);
    }

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
        .setInput({{0.0f, 1.0f, 2.0f, 3.0f, 4.0f},
                   {5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
                   {10.0f, 11.0f, 12.0f, 13.0f, 14.0f},
                   {15.0f, 16.0f, 17.0f, 18.0f, 19.0f},
                   {20.0f, 21.0f, 22.0f, 23.0f, 24.0f}});


    auto output = cnn.convolve(cnn.getInput());

    cout << "Convolution output:\n";
    printMatrix(cnn.getInput(), "Input");
    
    prinFeatures(output, "Convolution - Feature maps");

    output = cnn.applyNextLayer(output);
    prinFeatures(output, "ReLU activation - Feature maps");

    output = cnn.applyNextLayer(output);
    prinFeatures(output, "MaxPooling - Feature maps");

    output = cnn.applyNextLayer(output);
    auto flattenOutput = flatten.getFlattenedOutput();
    printFlattened(flattenOutput, "Flattened output");


    return 0;
}
