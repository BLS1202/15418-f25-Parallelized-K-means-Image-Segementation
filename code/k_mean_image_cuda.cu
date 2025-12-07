#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <ctime>
#include <string>
#include <limits>
#include <chrono>
#include <iomanip>
#include "libraries/Image.h"

#define THREAD_X 32
#define THREAD_Y 32
#define BLOCKSIZE (THREAD_X * THREAD_Y) // 1024
#define K 8

struct Point {
    float r, g, b;
};



// the squared distance between two colors.
__device__ float color_distance_sq(Point p1, Point p2) {
    float dr = p1.r - p2.r;
    float dg = p1.g - p2.g;
    float db = p1.b - p2.b;
    return dr*dr + dg*dg + db*db;
}

// kernel 1 assigns each pixel to its nearest centroid.
__global__ void assign_clusters_kernel(const Point* d_inputImage, int* d_clusterIds, const Point* d_centroids, int numPoints, int k, int width) {
    int pixelIndex = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    if (pixelIndex >= numPoints) return;

    Point pixelColor = d_inputImage[pixelIndex];
    float min_dist = 1e30f; // Use a very large number
    int best_centroid = 0;

    for (int i = 0; i < k; i++) {
        float dist = color_distance_sq(pixelColor, d_centroids[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = i;
        }
    }
    d_clusterIds[pixelIndex] = best_centroid;
}

// kernel 2 sums the colors and counts for each cluster using atomic operations.
__global__ void update_centroids_kernel(const Point* d_inputImage, const int* d_clusterIds, Point* d_sums, int* d_counts, int numPoints) {
    int pixelIndex = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pixelIndex >= numPoints) return;

    int clusterId = d_clusterIds[pixelIndex];
    Point pixelColor = d_inputImage[pixelIndex];

    atomicAdd(&(d_sums[clusterId].r), pixelColor.r);
    atomicAdd(&(d_sums[clusterId].g), pixelColor.g);
    atomicAdd(&(d_sums[clusterId].b), pixelColor.b);
    atomicAdd(&(d_counts[clusterId]), 1);
}

// kernel 3 calculates the new average color for each centroid.
__global__ void calculate_new_centroids_kernel(Point* d_centroids, const Point* d_sums, const int* d_counts, int k) {
    int centroidIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (centroidIndex >= k) return;

    int count = d_counts[centroidIndex];
    if (count > 0) {
        Point sum = d_sums[centroidIndex];
        d_centroids[centroidIndex].r = sum.r / count;
        d_centroids[centroidIndex].g = sum.g / count;
        d_centroids[centroidIndex].b = sum.b / count;
    }
}

// kernel 4 generates the final output image from the clustering results.
__global__ void generate_output_image_kernel(Point* d_outputImage, const int* d_clusterIds, const Point* d_centroids, int numPoints) {
    int pixelIndex = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    if (pixelIndex >= numPoints) return;
    
    int clusterId = d_clusterIds[pixelIndex];
    d_outputImage[pixelIndex] = d_centroids[clusterId];
}


int main(int argc, char** argv) {
    const auto init_start = std::chrono::steady_clock::now();
    int MAX_ITERATIONS = 20;

    int IMG_WIDTH = 0;
    int IMG_HEIGHT = 0;

    if(argc < 3){
        std::cerr << "Usage: " << argv[0] << "<input_image.jpg>" << "<output_image.jpg>" << std::endl;
        return 1;
    }

    std::string imagePath = argv[1];
    std::string outPath = argv[2];
    Image* originalImage = new Image(0, 0);
    if (!originalImage->loadJPG(imagePath)) {
        delete originalImage;
        return 1;
    }

    std::cout << "Starting K-Means Color Clustering..." << std::endl;
    std::cout << "  Clusters (K): " << K << std::endl;
    std::cout << "  Max Iterations: " << MAX_ITERATIONS << std::endl;
    std::cout << "------------------------------------" << std::endl;

    std::vector<Point> h_input_points;


    IMG_WIDTH = originalImage->width;   
    IMG_HEIGHT = originalImage->height; 

    float* img_data = originalImage -> data;
    size_t total_pixels = (size_t)IMG_WIDTH * IMG_HEIGHT;

    for (size_t i = 0; i < total_pixels; ++i) {

        size_t data_index = i * 4; 

        float r = (float)(img_data[data_index + 0] * 255.0f); 
        float g = (float)(img_data[data_index + 1] * 255.0f); 
        float b = (float)(img_data[data_index + 2] * 255.0f); 

        h_input_points.push_back(Point{r, g, b});
    }

    std::cout << "Loaded " << h_input_points.size() << " pixels as data points." << std::endl;

    int numPoints = IMG_WIDTH * IMG_HEIGHT;
    int imageBytes = numPoints * sizeof(Point);

    Point* d_inputImage;
    Point* d_outputImage;
    Point* d_centroids;
    Point* d_sums;
    int* d_clusterIds;
    int* d_counts;

    cudaMalloc(&d_inputImage, imageBytes);
    cudaMalloc(&d_outputImage, imageBytes);
    cudaMalloc(&d_centroids, K * sizeof(Point));
    cudaMalloc(&d_clusterIds, numPoints * sizeof(int));
    cudaMalloc(&d_sums, K * sizeof(Point));
    cudaMalloc(&d_counts, K * sizeof(int));

    // load input image to device
    cudaMemcpy(d_inputImage, h_input_points.data(), imageBytes, cudaMemcpyHostToDevice);

    // Initialize centroids by picking random pixels
    std::vector<Point> h_centroids(K);
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, numPoints - 1);
    
    for (int i = 0; i < K; i++) {
        h_centroids[i] = h_input_points[dist(rng)];
    }
    cudaMemcpy(d_centroids, h_centroids.data(), K * sizeof(Point), cudaMemcpyHostToDevice);

    // Setup Grid and Block dimensions
    int blockSize = BLOCKSIZE;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);
    
    dim3 centroidGridDim((K + 1023) / 1024);
    dim3 centroidBlockDim(1024);

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();


    std::cout << "Running K-Means algorithm on GPU..." << std::endl;
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        assign_clusters_kernel<<<gridDim, blockDim>>>(d_inputImage, d_clusterIds, d_centroids, numPoints, K, IMG_WIDTH);

        cudaMemset(d_sums, 0, K * sizeof(Point));
        cudaMemset(d_counts, 0, K * sizeof(int));
        
        update_centroids_kernel<<<gridDim, blockDim>>>(d_inputImage, d_clusterIds, d_sums, d_counts, numPoints);
        
        calculate_new_centroids_kernel<<<centroidGridDim, centroidBlockDim>>>(d_centroids, d_sums, d_counts, K);
    }
    cudaDeviceSynchronize();
    std::cout << "K-Means iterations complete." << std::endl;

    generate_output_image_kernel<<<gridDim, blockDim>>>(d_outputImage, d_clusterIds, d_centroids, numPoints);
    cudaDeviceSynchronize();

    std::cout << "------------------------------------" << std::endl;
    std::cout << "Generating quantized image data..." << std::endl;
    
    std::vector<Point> h_output_points(numPoints);
    cudaMemcpy(h_output_points.data(), d_outputImage, imageBytes, cudaMemcpyDeviceToHost);
    
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';


    std::vector<unsigned char> result_image(numPoints * 3);
    for (size_t i = 0; i < h_output_points.size(); ++i) {
        result_image[i * 3 + 0] = static_cast<unsigned char>(h_output_points[i].r);
        result_image[i * 3 + 1] = static_cast<unsigned char>(h_output_points[i].g);
        result_image[i * 3 + 2] = static_cast<unsigned char>(h_output_points[i].b);
    }
    std::cout << "Image data stored in vector." << std::endl;
    
    originalImage -> save_image_to_jpg(outPath, result_image, IMG_WIDTH, IMG_HEIGHT, 90);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_centroids);
    cudaFree(d_clusterIds);
    cudaFree(d_sums);
    cudaFree(d_counts);

    return 0;
}
