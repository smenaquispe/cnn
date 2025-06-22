// src/CNN.cpp
#include "CNN.h"
#include <iostream>

using namespace std;

vector<vector<float>> CNN::addPadding(const vector<vector<float>> &input)
{
    int input_h = input.size();
    int input_w = input[0].size();
    vector<vector<float>> padded(input_h + 2 * padding, vector<float>(input_w + 2 * padding, 0.0f));

    for (int i = 0; i < input_h; ++i)
    {
        for (int j = 0; j < input_w; ++j)
        {
            padded[i + padding][j + padding] = input[i][j];
        }
    }

    return padded;
}

vector<vector<vector<float>>> CNN::convolve(const vector<vector<float>> &input)
{
    vector<vector<float>> paddedInput = addPadding(input);
    int input_h = paddedInput.size();
    int input_w = paddedInput[0].size();
    int num_filters = filters.size();
    int fh = filters[0].getHeight();
    int fw = filters[0].getWidth();

    int out_h = (input_h - fh) / stride + 1;
    int out_w = (input_w - fw) / stride + 1;

    vector<vector<vector<float>>> output(num_filters, vector<vector<float>>(out_h, vector<float>(out_w, 0.0f)));

    for (int f = 0; f < num_filters; ++f)
    {
        for (int i = 0; i < out_h; ++i)
        {
            for (int j = 0; j < out_w; ++j)
            {
                float sum = 0.0f;
                for (int fi = 0; fi < fh; ++fi)
                {
                    for (int fj = 0; fj < fw; ++fj)
                    {
                        int x = i * stride + fi;
                        int y = j * stride + fj;
                        sum += paddedInput[x][y] * filters[f].getWeights()[fi][fj];
                    }
                }
                output[f][i][j] = sum;
            }
        }
    }

    return output;
}