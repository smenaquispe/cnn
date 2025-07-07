#ifndef CNN_H
#define CNN_H

#include <vector>
#include "Filter.h"
#include "Layer.h"
#include "Tensor.h"

class CNN
{
private:
    std::vector<Filter> filters;
    int stride;
    int padding;
    Tensor input;                      
    std::vector<Layer*> layers;
    int idxLayer = 0;

public:
    CNN() = default;

    // setters
    CNN &setFilters(const std::vector<Filter> &filters) {
        this->filters = filters;
        return *this;
    }

    CNN &setStride(int stride) {
        this->stride = stride;
        return *this;
    }

    CNN &setPadding(int padding) {
        this->padding = padding;
        return *this;
    }

    CNN &setInput(const Tensor &inputData) {
        this->input = inputData;
        return *this;
    }

    CNN &addLayer(Layer *layer) {
        layers.push_back(layer);
        return *this;
    }

    // getters
    std::vector<Filter> getFilters() const { return filters; }
    int getStride() const { return stride; }
    int getPadding() const { return padding; }
    Tensor getInput() const { return input; }

    // methods
    Tensor convolve(const Tensor &input);
    Tensor addPadding(const Tensor &input);
    Tensor applyLayers(const Tensor &input);
    Tensor applyNextLayer(const Tensor &input);
};

#endif // CNN_H
