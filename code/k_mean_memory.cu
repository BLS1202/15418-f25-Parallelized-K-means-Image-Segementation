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

#define THREAD_X 32
#define THREAD_Y 32
#define BLOCKSIZE (THREAD_X * THREAD_Y) // 1024
#define K 8

struct Point {
    float r, g, b;
};

// helper host
void save_image_to_ppm(const std::string& filename, const std::vector<unsigned char>& image_data, int width, int height) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
        return;
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image_data.data()), image_data.size());
    file.close();
    std::cout << "Successfully saved quantized image to '" << filename << "'" << std::endl;
}

// device
__device__ float color_distance_sq(Point p1, Point p2) {
    float dr = p1.r - p2.r;
    float dg = p1.g - p2.g;
    float db = p1.b - p2.b;
    return dr*dr + dg*dg + db*db;
}

// kernel 1 assignment and sum
__global__ void k_means_v2(Point* d_inputImage, Point* d_centroid_sums, 
    int* d_counts, Point* d_centroids, int numPoints, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelIndex = y * width + x;
    int thread_num = threadIdx.y * blockDim.x + threadIdx.x; // Thread ID within the block (0-1023)

    // Shared memory for this block's pixel data, cluster assignments, and partial sums
    __shared__ Point data_point[BLOCKSIZE];
    __shared__ int clusterIds[BLOCKSIZE];
    __shared__ Point partial_sums[K];
    __shared__ int partial_counts[K];
    __shared__ Point centroids_W[K];

    // Load this block's pixel data into shared memory
    if (pixelIndex < numPoints) {
        data_point[thread_num] = d_inputImage[pixelIndex];
    }

    // Initialize the shared memory for sums/counts for this block.
    if (thread_num < K) {
        partial_sums[thread_num] = {0.0f, 0.0f, 0.0f};
        partial_counts[thread_num] = 0;
        centroids_W[thread_num] = d_centroids[thread_num];
    }

    __syncthreads();

    // assignment
    if (pixelIndex < numPoints) {
        Point pixelColor = data_point[thread_num];
        float min_dist = 1e30f;
        int best_centroid_id = 0;

        for (int i = 0; i < K; i++) {
            float dist = color_distance_sq(pixelColor, centroids_W[i]);
            if (min_dist > dist) {
                min_dist = dist;
                best_centroid_id = i;
            }
        }

        atomicAdd(&partial_sums[best_centroid_id].r, data_point[thread_num].r);
        atomicAdd(&partial_sums[best_centroid_id].g, data_point[thread_num].g);
        atomicAdd(&partial_sums[best_centroid_id].b, data_point[thread_num].b);
        atomicAdd(&partial_counts[best_centroid_id], 1);

    }
    __syncthreads();

    // one thread: adds the block's total sums to the global memory sums.
    if (thread_num == 0) {
        for (int i = 0; i < K; i++) {
            if (partial_counts[i] > 0) {
                atomicAdd(&d_centroid_sums[i].r, partial_sums[i].r);
                atomicAdd(&d_centroid_sums[i].g, partial_sums[i].g);
                atomicAdd(&d_centroid_sums[i].b, partial_sums[i].b);
                atomicAdd(&d_counts[i], partial_counts[i]);
            }
        }
    }
}


// kernel 2 calculate new centroid averages
__global__ void update_centroids(Point* d_centroids, Point* d_centroid_sums, int* d_counts) {
    int i = threadIdx.x;
    if (i >= K) return;
    if (d_counts[i] > 0) {
        d_centroids[i].r = d_centroid_sums[i].r / d_counts[i];
        d_centroids[i].g = d_centroid_sums[i].g / d_counts[i];
        d_centroids[i].b = d_centroid_sums[i].b / d_counts[i];
    }
}

// kernal 3 get output image
__global__ void generate_output_image(Point* d_outputImage, Point* d_inputImage, Point* d_centroids, int numPoints, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_num = threadIdx.y * blockDim.x + threadIdx.x;
    int pixelIndex = y * width + x;

    __shared__ Point centroids_W[K];

    if (pixelIndex >= numPoints) return;

    // load the global centroids into shared memory.
    if (thread_num < K) {
        centroids_W[thread_num] = d_centroids[thread_num];
    }

    __syncthreads();
    
    Point pixelColor = d_inputImage[pixelIndex];
    float min_dist = 1e30f;
    int best_centroid_id = 0;

    for (int i = 0; i < K; i++) {
        float dist = color_distance_sq(pixelColor, centroids_W[i]);
        if (min_dist > dist) {
            min_dist = dist;
            best_centroid_id = i;
        }
    }
    d_outputImage[pixelIndex] = centroids_W[best_centroid_id];
}


// host
int main() {
    const auto init_start = std::chrono::steady_clock::now();
    int MAX_ITERATIONS = 20;

    int IMG_WIDTH = 0;
    int IMG_HEIGHT = 0;

    std::cout << "Starting K-Means Color Clustering..." << std::endl;
    std::cout << "  Clusters (K): " << K << std::endl;
    std::cout << "  Max Iterations: " << MAX_ITERATIONS << std::endl;
    std::cout << "------------------------------------" << std::endl;

    std::vector<Point> h_input_points;
    std::string inputFilename = "../img/camera_man.ppm";

    std::ifstream ppm_file(inputFilename, std::ios::in | std::ios::binary);
    if (!ppm_file) {
        std::cerr << "Error: Could not open file '" << inputFilename << "'. Please check the path." << std::endl;
        return 1;
    }

    std::string line;
    ppm_file >> line; 
    while (ppm_file.peek() == '\n' || ppm_file.peek() == '#') { ppm_file.ignore(256, '\n'); }
    ppm_file >> IMG_WIDTH >> IMG_HEIGHT;
    ppm_file.ignore(256, '\n');
    ppm_file.ignore(256, '\n');

    std::cout << "Reading image '" << inputFilename << "' (" << IMG_WIDTH << "x" << IMG_HEIGHT << ")" << std::endl;

    std::vector<unsigned char> raw_pixel_data(IMG_WIDTH * IMG_HEIGHT * 3);
    ppm_file.read(reinterpret_cast<char*>(raw_pixel_data.data()), raw_pixel_data.size());
    ppm_file.close();

    h_input_points.resize(IMG_WIDTH * IMG_HEIGHT);
    for (size_t i = 0; i < h_input_points.size(); ++i) {
        h_input_points[i] = {(float)raw_pixel_data[i*3], (float)raw_pixel_data[i*3+1], (float)raw_pixel_data[i*3+2]};
    }
    std::cout << "Loaded " << h_input_points.size() << " pixels as data points." << std::endl;

    int numPoints = IMG_WIDTH * IMG_HEIGHT;
    int imageBytes = numPoints * sizeof(Point);

    Point* d_inputImage;
    Point* d_outputImage;
    Point* d_centroids;
    Point* d_centroid_sums;
    int*   d_counts;

    cudaMalloc(&d_inputImage, imageBytes);
    cudaMalloc(&d_outputImage, imageBytes);
    cudaMalloc(&d_centroids, K * sizeof(Point));
    cudaMalloc(&d_centroid_sums, K * sizeof(Point));
    cudaMalloc(&d_counts, K * sizeof(int));

    cudaMemcpy(d_inputImage, h_input_points.data(), imageBytes, cudaMemcpyHostToDevice);

    std::vector<Point> h_centroids(K);
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, numPoints - 1);

    for (int i = 0; i < K; i++) {
        h_centroids[i] = h_input_points[dist(rng)];
    }
    cudaMemcpy(d_centroids, h_centroids.data(), K * sizeof(Point), cudaMemcpyHostToDevice);

    dim3 blockDim(THREAD_X, THREAD_Y);
    dim3 gridDim(((IMG_WIDTH)+THREAD_X-1)/THREAD_X, ((IMG_HEIGHT)+THREAD_Y-1)/THREAD_Y);

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();

    // k means
    std::cout << "Running K-Means algorithm on GPU..." << std::endl;
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        cudaMemset(d_centroid_sums, 0, K * sizeof(Point));
        cudaMemset(d_counts, 0, K * sizeof(int));

        k_means_v2<<<gridDim, blockDim>>>(d_inputImage, d_centroid_sums, d_counts, d_centroids, numPoints, IMG_WIDTH);
        cudaDeviceSynchronize();

        update_centroids<<<1, K>>>(d_centroids, d_centroid_sums, d_counts);
        cudaDeviceSynchronize();
    }
    std::cout << "K-Means iterations complete." << std::endl;
    std::cout << "------------------------------------" << std::endl;

    std::cout << "Generating quantized image data..." << std::endl;
    generate_output_image<<<gridDim, blockDim>>>(d_outputImage, d_inputImage, d_centroids, numPoints, IMG_WIDTH);

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

    save_image_to_ppm("kmeans_quantized.ppm", result_image, IMG_WIDTH, IMG_HEIGHT);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_centroids);
    cudaFree(d_centroid_sums);
    cudaFree(d_counts);

    return 0;
}
