// utils/printMatrix.h
#include <iostream>
#include <vector>
#include "Tensor.h"
using namespace std;

void printMatrix(const vector<vector<float>> &matrix, const string &title = "")
{
    if (!title.empty())
    {
        cout << "\n=== " << title << " ===\n";
    }

    for (const auto &row : matrix)
    {
        for (float value : row)
        {
            cout << value << "\t";
        }
        cout << "\n";
    }

    cout << "-----------------------------\n";
}

void printTensor(const Tensor &tensor, const string &title = "")
{
    if (!title.empty())
    {
        cout << "\n=== " << title << " ===\n";
    }

    auto shape = tensor.getShape();
    
    if (shape.size() == 2) {
        // 2D tensor: H x W
        size_t H = shape[0];
        size_t W = shape[1];
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {
                cout << tensor.at({i, j}) << "\t";
            }
            cout << "\n";
        }
    } else if (shape.size() == 3) {
        // 3D tensor: C x H x W
        size_t C = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        for (size_t c = 0; c < C; ++c) {
            cout << "Channel " << c + 1 << ":\n";
            for (size_t i = 0; i < H; ++i) {
                for (size_t j = 0; j < W; ++j) {
                    cout << tensor.at({c, i, j}) << "\t";
                }
                cout << "\n";
            }
            cout << "\n";
        }
    } else if (shape.size() == 1) {
        // 1D tensor: flattened
        size_t size = shape[0];
        for (size_t i = 0; i < size; ++i) {
            cout << tensor.at({i}) << "\t";
        }
        cout << "\n";
    }

    cout << "-----------------------------\n";
}

void prinFeatures(const vector<vector<vector<float>>> &features, const string &title = "")
{
    if (!title.empty())
    {
        cout << "\n=== " << title << " ===\n";
    }

    for (size_t i = 0; i < features.size(); ++i)
    {
        cout << "Feature map " << i + 1 << ":\n";
        printMatrix(features[i]);
    }
}

void printFlattened(const vector<float> &flattened, const string &title = "")
{
    if (!title.empty())
    {
        cout << "\n=== " << title << " ===\n";
    }

    for (const auto &value : flattened)
    {
        cout << value << "\t";
    }
    cout << "\n-----------------------------\n";
}

void printFlattenedTensor(const Tensor &tensor, const string &title = "")
{
    if (!title.empty())
    {
        cout << "\n=== " << title << " ===\n";
    }

    auto shape = tensor.getShape();
    if (shape.size() != 1) {
        cout << "Warning: Tensor is not 1D, but treating as flattened.\n";
    }

    for (const auto &value : tensor.data)
    {
        cout << value << "\t";
    }
    cout << "\n-----------------------------\n";
}