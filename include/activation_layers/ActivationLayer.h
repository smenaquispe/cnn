// include/ActivationLayer.h
#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "Layer.h"
#include <vector>

using namespace std;

class ActivationLayer : public Layer
{
public:
    virtual ~ActivationLayer() = default;
};

#endif // ACTIVATION_LAYER_H