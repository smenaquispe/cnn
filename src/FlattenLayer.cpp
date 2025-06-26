#include "FlattenLayer.h"

 vector<vector<vector<float>>> FlattenLayer::apply(const vector<vector<vector<float>>> &input)
{
    flattenedOutput.clear();
    for (const auto &channel : input)
    {
        for (const auto &row : channel)
        {
            for (float val : row)
            {
                flattenedOutput.push_back(val);
            }
        }
    }

    return {{{}}}; 
}

void FlattenLayer::apply2D(const vector<vector<float>> &input)
{
    flattenedOutput.clear();
    for (const auto &row : input)
    {
        for (float val : row)
        {
            flattenedOutput.push_back(val);
        }
    }
}

const vector<float> &FlattenLayer::getFlattenedOutput() const
{
    return flattenedOutput;
}