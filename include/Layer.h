// include/layers/Layer.h
#ifndef LAYER_H
#define LAYER_H

#include <vector>

using namespace std;

class Layer
{
public:
    virtual ~Layer() = default;

    virtual vector<vector<vector<float>>> apply(const vector<vector<vector<float>>> &input) = 0;
};

#endif // LAYER_H
