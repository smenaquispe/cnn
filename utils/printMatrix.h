// utils/printMatrix.h
#include <iostream>
#include <vector>
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