#include "Filter.h"
#include <cstdlib>
#include <ctime>
#include <stdexcept>

Filter &Filter::initWeights(const std::vector<size_t> &shape)
{
    weights.shape = shape;
    size_t total = weights.totalSize();
    weights.data.resize(total);

    for (size_t i = 0; i < total; ++i)
    {
        weights.data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    return *this;
}

Filter &Filter::initWeights()
{
    if (width == 0 || height == 0) {
        throw std::invalid_argument("Width and height must be set before initializing weights");
    }
    
    return initWeights({height, width});
}
