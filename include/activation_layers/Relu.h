// include/ReLU.h
#ifndef RELU_H
#define RELU_H

#include "ActivationLayer.h"
#include <algorithm>

class ReLU : public ActivationLayer
{
public:
   ~ReLU() override = default;
   vector<vector<vector<float>>> apply(const vector<vector<vector<float>>> &input) override;
};

#endif // RELU_H
