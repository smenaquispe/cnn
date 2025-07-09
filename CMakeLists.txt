cmake_minimum_required(VERSION 3.10)

project(cnn VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluir todos los headers
include_directories(include)

# Agregar el ejecutable original con TODOS los .cpp necesarios
add_executable(cnn
    main.cpp
    src/CNN.cpp
    src/Filter.cpp
    src/activation_layers/Relu.cpp
    src/pooling_layers/MaxPooling.cpp
    src/FlattenLayer.cpp
)

# Agregar el ejecutable de entrenamiento
add_executable(cnn_training
    main_training.cpp
    src/NeuralNetwork.cpp
    src/DenseLayer.cpp
    src/Loss.cpp
    src/DataGenerator.cpp
    src/Filter.cpp
    src/activation_layers/Relu.cpp
    src/activation_layers/Softmax.cpp
    src/pooling_layers/MaxPooling.cpp
    src/FlattenLayer.cpp
)

# Agregar el ejecutable de ejemplo
add_executable(example_usage
    example_usage.cpp
    src/NeuralNetwork.cpp
    src/DenseLayer.cpp
    src/Loss.cpp
    src/DataGenerator.cpp
    src/Filter.cpp
    src/activation_layers/Relu.cpp
    src/activation_layers/Softmax.cpp
    src/pooling_layers/MaxPooling.cpp
    src/FlattenLayer.cpp
)
