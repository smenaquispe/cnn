// include/CNN.h
#ifndef CNN_H
#define CNN_H

#include <vector>
#include "Filter.h"
#include "Layer.h"

using namespace std;

class CNN
{
private:
    vector<Filter> filters;
    int stride;
    int padding;
    vector<vector<float>> input;
    vector<Layer*> layers;

    int idxLayer = 0;

public:
    CNN() = default;

    // setters
    CNN &setFilters(const vector<Filter> &filters)
    {
        this->filters = filters;
        return *this;
    }
    CNN &setStride(int stride)
    {
        this->stride = stride;
        return *this;
    }
    CNN &setPadding(int padding)
    {
        this->padding = padding;
        return *this;
    }
    CNN &setInput(const vector<vector<float>> &inputData)
    {
        this->input = inputData;
        return *this;
    }
    CNN &addLayer(Layer *layer)
    {
        layers.push_back(layer);
        return *this;
    }

    // getters
    vector<Filter> getFilters() const { return filters; }
    int getStride() const { return stride; }
    int getPadding() const { return padding; }
    vector<vector<float>> getInput() const { return input; }

    // methods
    vector<vector<vector<float>>> convolve(const vector<vector<float>> &input);
    vector<vector<float>> addPadding(const vector<vector<float>> &input);
    vector<vector<vector<float>>> applyLayers(const vector<vector<vector<float>>> &input);
    vector<vector<vector<float>>> applyNextLayer(const vector<vector<vector<float>>> &input);
    
};

#endif // CNN_H