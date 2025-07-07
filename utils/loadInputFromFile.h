// utils/loadInputFromFile.h
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <string>
#include "Tensor.h"

using namespace std;

vector<vector<float>> loadInputFromFile(const string& filename) {
    vector<vector<float>> input;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo: " << filename << endl;
        return input;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<float> row;
        float value;
        while (iss >> value) {
            row.push_back(value);
        }
        input.push_back(row);
    }

    file.close();
    return input;
}

Tensor loadInputFromFileAsTensor(const string& filename) {
    vector<vector<float>> input = loadInputFromFile(filename);
    
    if (input.empty()) {
        return Tensor(); // Return empty tensor on error
    }

    size_t H = input.size();
    size_t W = input[0].size();
    
    Tensor tensor;
    tensor.shape = {H, W};
    tensor.data.resize(tensor.totalSize());
    
    for (size_t i = 0; i < H; ++i) {
        for (size_t j = 0; j < W; ++j) {
            tensor.at({i, j}) = input[i][j];
        }
    }
    
    return tensor;
}
