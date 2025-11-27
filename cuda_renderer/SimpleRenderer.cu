#include <iostream>
#include <cuda_runtime.h>
#include "SimpleRenderer.h"
#include "Image.h"
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <ctime>
#include <limits>

// =================================================================================
// CUDA Kernel (Device Code)
// =================================================================================

// This is the simplest possible "renderer". Each thread is responsible for one
// pixel. It just copies the color from an input buffer to an output buffer.
__global__ void kernel_copy_image(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // int offset = 4 * (y * width + x);

    // // Using float4 is an optimization that tells the compiler to use a single
    // // 128-bit memory transaction instead of four 32-bit ones.
    // ((float4*)output)[y * width + x] = ((float4*)input)[y * width + x];



    //image flipped back version:
    // Read from the original y coordinate
    int input_index = y * width + x;
    
    // Calculate the flipped y coordinate for writing
    int flipped_y = height - 1 - y;
    int output_index = flipped_y * width + x;
    // Copy from the input location to the flipped output location
    ((float4*)output)[output_index] = ((float4*)input)[input_index];
}

// =================================================================================
// C++ Class Implementation (Host Code)
// =================================================================================


struct Point {
    float r, g, b, alpha; // Using double for precision in centroid calculations
    int clusterId;
};

float distance(Point p1, Point p2) {
    return std::sqrt(std::pow(p1.r - p2.r, 2) 
        + std::pow(p1.g - p2.g, 2) 
        + std::pow(p1.b - p2.b, 2)
        + std::pow(p1.alpha - p2.alpha, 2));
}


SimpleRenderer::SimpleRenderer(Image* image) {
    width = image->width;
    height = image->height;
    
    // Allocate a new image on the host for displaying the final result
    displayImage = new Image(width, height);

    size_t imageBytes = sizeof(float) * 4 * width * height;

    std::cout << "Allocating memory on GPU..." << std::endl;
    
    // 1. Allocate memory on the GPU device
    cudaMalloc(&d_inputImage, imageBytes);
    cudaMalloc(&d_outputImage, imageBytes);

    // 2. Copy the input image from host (CPU) to device (GPU)
    cudaMemcpy(d_inputImage, image->data, imageBytes, cudaMemcpyHostToDevice);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after setup: %s\n", cudaGetErrorString(err));
    }
}

SimpleRenderer::~SimpleRenderer() {
    delete displayImage;
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void SimpleRenderer::render() {
    // Configure the kernel launch grid
    // Use 16x16 threads per block (256 total)
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);

    // 3. Launch the kernel on the GPU
    kernel_copy_image<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height);
    
    // Wait for the kernel to finish before proceeding
    cudaDeviceSynchronize();
}

const Image* SimpleRenderer::getDisplayImage() {
    size_t imageBytes = sizeof(float) * 4 * width * height;
    
    // 4. Copy the final rendered image from device (GPU) back to host (CPU)
    cudaMemcpy(displayImage->data, d_outputImage, imageBytes, cudaMemcpyDeviceToHost);

    return displayImage;
}

Image* SimpleRenderer::segmentationImage(const Image* inputImage){
    std::cout << "Processing image..." << std::endl;
    int IMG_WIDTH = 0;
    int IMG_HEIGHT = 0;
    
    // >> UNCHANGED: These can still be configured. <<
    const int K = 8; // Number of clusters (dominant colors)
    const int MAX_ITERATIONS = 20;
    IMG_WIDTH = inputImage -> width;
    IMG_HEIGHT = inputImage -> height;
    std::vector<Point> points;
    size_t imageBytes = sizeof(float) * 4 * width * height;
    
    float* copied_data = new float[IMG_WIDTH * IMG_HEIGHT * 4];
    memcpy(copied_data, inputImage->data, imageBytes);

    for(size_t i = 0; i < IMG_WIDTH * IMG_HEIGHT * 4; i+=4){
        points.push_back({copied_data[i], 
                        copied_data[i+1], 
                        copied_data[i+2],
                        copied_data[i+3],
                        -1});
        
    }
    std::cout << "Loaded " << points.size() << " pixels as data points." << std::endl;

    //initialize centroids
    std::vector<Point> centroids;
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, points.size() - 1);
    for (int i = 0; i < K; ++i) {
        centroids.push_back(points[dist(rng)]);
    }

    //run k-means

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        bool changed = false;
        // Assignment Step
        for (auto& point : points) {
            float min_dist = std::numeric_limits<float>::max();
            int closest_centroid_id = -1;
            for (int i = 0; i < K; ++i) {
                double d = distance(point, centroids[i]);
                if (d < min_dist) { min_dist = d; closest_centroid_id = i; }
            }
            if (point.clusterId != closest_centroid_id) {
                point.clusterId = closest_centroid_id;
                changed = true;
            }
        }
        // Update Step
        std::vector<Point> new_centroids(K, {0, 0, 0, -1}); // >> Changed to {r,g,b,id}
        std::vector<int> counts(K, 0);
        for (const auto& point : points) {
            int cluster_id = point.clusterId;
            new_centroids[cluster_id].r += point.r; // >> Changed from .x to .r
            new_centroids[cluster_id].g += point.g; // >> Changed from .y to .g
            new_centroids[cluster_id].b += point.b; // >> ADDED for blue channel <<
            new_centroids[cluster_id].alpha += point.alpha;
            counts[cluster_id]++;
        }
        for (int i = 0; i < K; ++i) {
            if (counts[i] > 0) {
                centroids[i].r = new_centroids[i].r / counts[i]; // >> Changed from .x to .r
                centroids[i].g = new_centroids[i].g / counts[i]; // >> Changed from .y to .g
                centroids[i].b = new_centroids[i].b / counts[i]; // >> ADDED for blue channel <<
                centroids[i].alpha = new_centroids[i].alpha /counts[i];
            }
        }
        // Convergence Check
        if (!changed) {
            std::cout << "Convergence reached at iteration " << iter + 1 << std::endl;
            break;
        } else {
            std::cout << "Iteration " << iter + 1 << " complete." << std::endl;
        }
    }

    std::cout << "Generating final image from cluster data..." << std::endl;
    // Create a new image to store the result.
    Image* outputImage = new Image(IMG_WIDTH, IMG_HEIGHT);
    // Loop through each original pixel's corresponding point.
    for (size_t i = 0; i < points.size(); ++i) {
        const Point& centroid_color = centroids[points[i].clusterId];
        
        int baseIndex = i * 4;
        
        outputImage->data[baseIndex + 0] = centroid_color.r;     // Red
        outputImage->data[baseIndex + 1] = centroid_color.g;     // Green
        outputImage->data[baseIndex + 2] = centroid_color.b;     // Blue
        outputImage->data[baseIndex + 3] = centroid_color.alpha; // Alpha
    }   
    std::cout << "Image processing complete." << std::endl;
    return outputImage;
}

Image* SimpleRenderer::segmentationImageCUDA(const Image* inputImage){
    std::cout << "Processing image with CUDA..." << std::endl;
    return NULL;
}
