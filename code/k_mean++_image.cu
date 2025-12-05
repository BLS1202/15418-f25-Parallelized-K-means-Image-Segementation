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
#include <numeric> // Added for std::accumulate

#define THREAD_X 32
#define THREAD_Y 32
#define BLOCKSIZE (THREAD_X * THREAD_Y) // 1024
#define K 8
// A smaller block size is often better for 1D reduction kernels
#define REDUCTION_BLOCK_SIZE 256

struct Point {
    float r, g, b;
};

// =================================================================================
// Helper Function (Host)
// =================================================================================

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

// =================================================================================
// CUDA Kernels and Device Functions
// =================================================================================

__device__ float color_distance_sq(Point p1, Point p2) {
    float dr = p1.r - p2.r;
    float dg = p1.g - p2.g;
    float db = p1.b - p2.b;
    return dr*dr + dg*dg + db*db;
}


// =================================================================================
// K-MEANS++ INITIALIZATION KERNELS (EDITED SECTION)
// =================================================================================

/**
 * K-Means++ Step 1: For each point, find the squared distance to the nearest existing centroid.
 */
__global__ void kmeans_pp_init_p1(double* d_distances, const Point* d_inputImage, const Point* d_centroids,
                                  int numPoints, int width, int centroid_size){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelIndex = y * width + x;

    if(pixelIndex >= numPoints) return;

    Point pixel_data = d_inputImage[pixelIndex];
    double min_d_sq = 1e30f;

    for(int i = 0; i < centroid_size; i++){
        double d_sq = color_distance_sq(pixel_data, d_centroids[i]);
         if(d_sq < min_d_sq){
            min_d_sq = d_sq;
         }
    }
    d_distances[pixelIndex] = min_d_sq;
}

/**
 * K-Means++ Step 2: Perform a parallel reduction to get the sum of distances for each block.
 */
__global__ void reduction(const double* d_distances, double* d_partial_sum, int numPoints){
    extern __shared__ double s_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    s_data[tid] = (i < numPoints) ? d_distances[i] : 0.0;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        d_partial_sum[blockIdx.x] = s_data[0];
    }
}

/**
 * K-Means++ Step 3: Find the new centroid by scanning within the chosen "winning" block.
 */
__global__ void kmeans_pp_init_p2(const double* d_distances, int* d_new_index, int numPoints, int search_block, double threshold){
    int start_index = search_block * REDUCTION_BLOCK_SIZE;
    int end_index = min((start_index + REDUCTION_BLOCK_SIZE), numPoints);

    double cumulative = 0.0;
    for(int i = start_index; i < end_index; i++){
        cumulative += d_distances[i];
        if(cumulative >= threshold){
            *d_new_index = i;
            return;
        }
    }
    *d_new_index = end_index - 1; // Failsafe
}


// =================================================================================
// CLUSTER PART (UNCHANGED SECTION)
// =================================================================================

__global__ void k_means_v2(Point* d_inputImage, Point* d_centroid_sums, 
    int* d_counts, Point* d_centroids, int numPoints, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelIndex = y * width + x;
    int thread_num = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ Point data_point[BLOCKSIZE];
    __shared__ int clusterIds[BLOCKSIZE];
    __shared__ Point partial_sums[K];
    __shared__ int partial_counts[K];
    __shared__ Point centroids_W[K];

    if (pixelIndex < numPoints) {
        data_point[thread_num] = d_inputImage[pixelIndex];
    }
    if (thread_num < K) {
        partial_sums[thread_num] = {0.0f, 0.0f, 0.0f};
        partial_counts[thread_num] = 0;
        centroids_W[thread_num] = d_centroids[thread_num];
    }
    __syncthreads();

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
        clusterIds[thread_num] = best_centroid_id;
    }
    __syncthreads();

    if (pixelIndex < numPoints) {
        int clusterId = clusterIds[thread_num];
        atomicAdd(&partial_sums[clusterId].r, data_point[thread_num].r);
        atomicAdd(&partial_sums[clusterId].g, data_point[thread_num].g);
        atomicAdd(&partial_sums[clusterId].b, data_point[thread_num].b);
        atomicAdd(&partial_counts[clusterId], 1);
    }
    __syncthreads();

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

__global__ void update_centroids(Point* d_centroids, Point* d_centroid_sums, int* d_counts) {
    int i = threadIdx.x;
    if (i >= K) return;
    if (d_counts[i] > 0) {
        d_centroids[i].r = d_centroid_sums[i].r / d_counts[i];
        d_centroids[i].g = d_centroid_sums[i].g / d_counts[i];
        d_centroids[i].b = d_centroid_sums[i].b / d_counts[i];
    }
}

__global__ void generate_output_image(Point* d_outputImage, Point* d_inputImage, Point* d_centroids, int numPoints, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_num = threadIdx.y * blockDim.x + threadIdx.x;
    int pixelIndex = y * width + x;

    __shared__ Point centroids_W[K];
    if (pixelIndex >= numPoints) return;
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


// =================================================================================
// Host Code (main function)
// =================================================================================

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

    std::vector<Point> h_centroids;
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, numPoints - 1);

    // =================================================================================
    // K-MEANS++ INITIALIZATION LOGIC (EDITED SECTION)
    // =================================================================================
    
    // Choose the first centroid uniformly at random
    int first_idx = dist(rng);
    h_centroids.push_back(h_input_points[first_idx]);
    
    // Allocate memory for initialization steps
    double* d_distances;
    double* d_partial_sum;
    int* d_new_index;
    cudaMalloc(&d_distances, numPoints * sizeof(double));
    cudaMalloc(&d_new_index, sizeof(int));

    // Configure launch parameters
    dim3 gridDim(((IMG_WIDTH)+THREAD_X-1)/THREAD_X, ((IMG_HEIGHT)+THREAD_Y-1)/THREAD_Y);
    dim3 blockDim(THREAD_X, THREAD_Y);
    int reductionGridSize = (numPoints + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    cudaMalloc(&d_partial_sum, reductionGridSize * sizeof(double));
    std::vector<double> h_partial_sums(reductionGridSize);

    // Loop to choose the remaining K-1 centroids
    while(h_centroids.size() < K){
        int centroid_size = h_centroids.size();
        cudaMemcpy(d_centroids, h_centroids.data(), centroid_size * sizeof(Point), cudaMemcpyHostToDevice);

        kmeans_pp_init_p1<<<gridDim, blockDim>>>(d_distances, d_inputImage, d_centroids, numPoints, IMG_WIDTH, centroid_size);
        
        size_t shared_mem_size = REDUCTION_BLOCK_SIZE * sizeof(double);
        reduction<<<reductionGridSize, REDUCTION_BLOCK_SIZE, shared_mem_size>>>(d_distances, d_partial_sum, numPoints);
        
        cudaMemcpy(h_partial_sums.data(), d_partial_sum, reductionGridSize * sizeof(double), cudaMemcpyDeviceToHost);
        double total_sum = std::accumulate(h_partial_sums.begin(), h_partial_sums.end(), 0.0);

        // Generate threshold and find the winning block
        std::uniform_real_distribution<double> threshold_dist(0.0, total_sum);
        double threshold = threshold_dist(rng);
        
        int winning_block = -1;
        double cumulative_sum = 0.0;
        for (int i = 0; i < reductionGridSize; ++i) {
            cumulative_sum += h_partial_sums[i];
            if (cumulative_sum >= threshold) {
                winning_block = i;
                threshold -= (cumulative_sum - h_partial_sums[i]); // Adjust threshold
                break;
            }
        }
        if (winning_block == -1) winning_block = reductionGridSize - 1;

        kmeans_pp_init_p2<<<1, 1>>>(d_distances, d_new_index, numPoints, winning_block, threshold);

        // Get the new index and add the point to our centroid list
        int new_idx;
        cudaMemcpy(&new_idx, d_new_index, sizeof(int), cudaMemcpyDeviceToHost);
        h_centroids.push_back(h_input_points[new_idx]);
    }

    // Free temporary memory used for initialization
    cudaFree(d_distances);
    cudaFree(d_partial_sum);
    cudaFree(d_new_index);

    cudaMemcpy(d_centroids, h_centroids.data(), K * sizeof(Point), cudaMemcpyHostToDevice);

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();

    // --- MAIN K-MEANS LOOP (UNCHANGED) ---
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
