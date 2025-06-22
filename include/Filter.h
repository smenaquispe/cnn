#ifndef FILTER_H
#define FILTER_H

#include <vector>
using namespace std;

class Filter
{
private:
    int width;
    int height;
    vector<vector<float>> weights;

public:
    Filter() = default;

    // setters
    Filter &setWidth(int w)
    {
        width = w;
        return *this;
    }
    Filter &setHeight(int h)
    {
        height = h;
        return *this;
    }
    Filter &setWeights(const vector<vector<float>> &w)
    {
        weights = w;
        return *this;
    }

    // getters
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    const vector<vector<float>> &getWeights() const { return weights; }

    // methods
    Filter &initWeights();
};

#endif // FILTER_H