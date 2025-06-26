#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <string>

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
