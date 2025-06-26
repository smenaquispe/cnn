#include "pooling_layers/AveragePooling.h"

vector<vector<vector<float>>> AveragePooling::apply(const vector<vector<vector<float>>> &input)
{
    vector<vector<vector<float>>> output;

    for (const auto &channel : input)
    {
        auto padded = pad(channel);
        int H = padded.size();
        int W = padded[0].size();

        int outH = (H - poolHeight) / stride + 1;
        int outW = (W - poolWidth) / stride + 1;

        vector<vector<float>> pooled(outH, vector<float>(outW, 0.0f));

        for (int i = 0; i < outH; ++i)
        {
            for (int j = 0; j < outW; ++j)
            {
                float sum = 0.0f;
                for (int m = 0; m < poolHeight; ++m)
                {
                    for (int n = 0; n < poolWidth; ++n)
                    {
                        int y = i * stride + m;
                        int x = j * stride + n;
                        sum += padded[y][x];
                    }
                }
                pooled[i][j] = sum / (poolHeight * poolWidth);
            }
        }

        output.push_back(pooled);
    }

    return output;
}