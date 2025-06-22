cmake_minimum_required(VERSION 3.10)

project(cnn VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(cnn
    main.cpp
    src/Filter.cpp
    src/CNN.cpp
)
