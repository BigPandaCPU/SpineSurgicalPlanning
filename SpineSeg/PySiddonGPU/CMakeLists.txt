# project(PySiddonGpu)
# add_compile_options(-std=c++11)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
# set(CMAKE_BUILD_TYPE "Release")
# set(PYBIND11_CPP_STANDARD -std=c++11)

# set(pybind11_DIR "C:/ProgramData/Anaconda3/Lib/site-packages/pybind11/share/cmake/pybind11")
# find_package(pybind11 REQUIRED)

# find_package(CUDA REQUIRED)

# if (CUDA_FOUND)
  # MESSAGE(STATUS "${CUDA_INCLUDE_LIBRARY}")
  # MESSAGE(STATUS "cuda found")
  # #INCLUDE_DIRECTORIES("/usr/local/cuda/include/")
  # #link_directories("/usr/local/cuda/lib64")

  # include_directories(${CUDA_INCLUDE_DIRS})
  # link_directories($ENV{CUDA_PATH}/lib/x64)
  
  # #MESSAGE(CUDA_DIR)
# endif()

# INCLUDE_DIRECTORIES(${pybind11_INCLUDE_DIRS})
# add_library(PySiddonGpu MODULE src/pywrapper.cpp src/siddon_class.cu src/siddon_class.cuh)
# target_link_libraries(PySiddonGpu pybind11::module cudart cudadevrt)


cmake_minimum_required(VERSION 3.18.2)
project(PySiddonGpu)
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(CMAKE_BUILD_TYPE "Release")
set(PYBIND11_CPP_STANDARD -std=c++11)
#set(pybind11_DIR "C:/ProgramData/Anaconda3/Lib/site-packages/pybind11/share/cmake/pybind11")
set(pybind11_DIR "/home/alg/miniconda3/lib/python3.7/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
INCLUDE_DIRECTORIES(${pybind11_INCLUDE_DIRS}
./)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
  MESSAGE(STATUS "${CUDA_INCLUDE_LIBRARY}")
  MESSAGE(STATUS "cuda found")
  INCLUDE_DIRECTORIES("/usr/local/cuda/include/")
  link_directories("/usr/local/cuda/lib64")

  #include_directories(${CUDA_INCLUDE_DIRS})
  #link_directories($ENV{CUDA_PATH}/lib/x64)
  
  #MESSAGE(CUDA_DIR)
endif()

cuda_add_library(PySiddonGpu MODULE src/pywrapper.cpp src/siddon_class.cu src/siddon_class.cuh src/book.h)

target_link_libraries(PySiddonGpu pybind11::module cudart cudadevrt)