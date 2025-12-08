# Parallelized-K-means-Image-Segementation (15418 Fall 2025 Project)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Description

 We implemented K-means clustering image segmentation in CUDA on GPU and in OpenMP on the CPU and compared the performance of non-optimized versions with these two parallelized versions. Our implementation demonstrated significant speedups. We also parallelized the initialization step of K-means++ in CUDA, and implemented a CUDA based image renderer that can run CUDA and OpenMP implementations of K-means clustering segmentation on image for every frame.

![Screenshot of the project](src/image.png)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Features](#features)
- [Usage](#usage)
- [Examples] (#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

-   **K-means Clustering:** Segments images by partitioning pixels into K clusters based on color.
-   **CPU Parallelization:** Utilizes OpenMP to accelerate the clustering algorithm on multi-core CPUs.
-   **GPU Parallelization:** Implements a highly parallel version of the algorithm using CUDA to leverage the power of NVIDIA GPUs.
-   **K-means++ Initialization:** Includes a parallelized implementation of the K-means++ algorithm in CUDA to select better initial cluster centroids and improve convergence.
-   **Image Segmentation Renderer:** A real-time renderer built with CUDA that can run and visualize the segmentation process for sequential, OpenMP, and CUDA implementations frame-by-frame.

## Prerequisites

    We used NVIDIA GeForce RTX 2080 B GPUs for implementations and testing. Any platform
    with NVIDIA GPU should work. Make sure the platform has NVIDIA CUDA Toolkit (including the nvcc compiler)

    If running OpenMP implementation, check how many cores are available on the CPU platform
    A C++ compiler with OpenMP support (e.g., modern GCC).

## Usage

1.  **Running the Renderer**

    Clone the respository

    ```bash
    git clone https://github.com/BLS1202/Parallelized-K-means-Image-Segementation.git
    ```

    To run the renderer:

    ```bash
    cd cuda_renderer
    make    
    ./cuda_renderer/kmeans_visualizer <mode> <path/to/image.jpg>
    ```

    For <mode> select between "simple", "cuda" and "openmp"
    simple: Runs the sequential CPU version.
    cuda: Runs the CUDA GPU version.
    openmp: Runs the OpenMP parallel CPU version

2. **Running Testing Programs**

    Programs that test performance are in the code directory

    Here is an example to run a simple CPU version:

    ```bash
    cd code
    g++ -c k_mean_image.cpp libraries/Image.cpp
    g++ k_mean_image.o Image.o -o k_means_normal
    .k_means_normal ./img/input.jpg ./img/output_cpu.jpg
    ```

    To run Cuda files:

    ```bash
    cd code
    g++ -c libraries/Image.cpp -o Image.o
    nvcc -c k_mean_memory.cu -o k_mean_memory.o
    nvcc Image.o k_mean_memory.o -o k_means_memory

    ./k_means_memory <path/to/input_image.jpg> <path/to/output_image.jpg>
    ```

    To run OpenMP implementations:

    ```bash
    cd code
    g++ -c libraries/Image.cpp -o Image.o
    g++ -c k_mean_image_openmp1.cpp -fopenmp
    g++ Image.o k_mean_image_openmp1.o -fopenmp -o k_mean_image_openmp1

    ./k_mean_image_openmp1 <num_threads> <path/to/input_image.jpg> <path/to/output_image.jpg>
    ```

    More details for compile and running the files can be found in compile_commands.txt

    


