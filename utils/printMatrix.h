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