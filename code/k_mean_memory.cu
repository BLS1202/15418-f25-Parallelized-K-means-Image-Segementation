// #include <iostream>
// #include <cuda_runtime.h>
// #include <vector>
// #include <fstream>
// #include <cmath>
// #include <random>
// #include <ctime>
// #include <string>
// #include <limits>
// #include <chrono>
// #include <iomanip>

// struct Point {
//     float r, g, b;
// };

// // =================================================================================
// // Helper Function (for saving the output)
// // =================================================================================

// // Error 8: Added the missing save_image_to_ppm function
// void save_image_to_ppm(const std::string& filename, const std::vector<unsigned char>& image_data, int width, int height) {
//     std::ofstream file(filename, std::ios::out | std::ios::binary);
//     if (!file) {
//         std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
//         return;
//     }
//     file << "P6\n" << width << " " << height << "\n255\n";
//     file.write(reinterpret_cast<const char*>(image_data.data()), image_data.size());
//     file.close();
//     std::cout << "Successfully saved quantized image to '" << filename << "'" << std::endl;
// }

// // =================================================================================
// // CUDA Kernel (Device Code)
// // =================================================================================

// /**
//  * A device helper function to calculate the squared distance between two colors.
//  */
// __device__ float color_distance_sq(Point p1, Point p2) {
//     float dr = p1.r - p2.r;
//     float dg = p1.g - p2.g;
//     float db = p1.b - p2.b;
//     return dr*dr + dg*dg + db*db;
// }

// __global__ void k_means_v2(Point* d_inputImage, Point* d_outputImage, 
//                             Point* d_centroids, int numPoints, int K){

//     int pixelIndex = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
//     if(pixelIndex >= numPoints){
//         return
//     }
//     int thread_num = threadIdx.x;

//     //use shared memory to store centroids and points information
//     __shared__ Point centroids_W[K];
//     __shared__ int counts[K];
//     __shared__ int clusterIds[blockDim.x];
//     __shared__ Point data_point[blockDim.x];


//     if(pixelIndex < K){
//         centroids_W[pixelIndex] = d_centroids[pixelIndex];
//         counts[pixelIndex] = 0
//     }

//     Point pixelColor = d_inputImage[pixelIndex];
//     float min_dist = 1e30f; // Use a very large number
//     int best_centroid = 0;

//     for(int i = 0; i < K; i++){
//         float dist = color_distance_sq(pixelColor, centroids_W[i]);
//         if(min_dist > dist){
//             min_dist = dist;
//             best_centroid = i;
//         }
//     }
    
//     __syncthreads();
//     //reduction on summing

//     data_point[thread_num] = pixelColor;
//     clusterIds[thread_num] = best_centroid;

//     __syncthreads();

//     if(thread_num == 0){
//         Point tmp_datapoints[K] = {0};
//         int tmp_cluster_counts[K] = {0};
    

//         for(int j = 0; j < blockDim.x; j++){
//             int clusterId = clusterIds[j];
//             tmp_datapoints[clusterId].r += data_point[j].r;
//             tmp_datapoints[clusterId].g += data_point[j].g;
//             tmp_datapoints[clusterId].b += data_point[j].b;
//             tmp_cluster_counts[clusterId] += 1;
//         }

//         for(int z = 0; z < K; z++){
//             atomicAdd(&centroids_W[z].r, tmp_datapoints[z].r);
//             atomicAdd(&centroids_W[z].g, tmp_datapoints[z].g);
//             atomicAdd(&centroids_W[z].b, tmp_datapoints[z].b);
//             atomicAdd(&counts[z], tmp_cluster_counts[z]);
//         }

//     }   

//     __syncthreads();

//     //calculate mean
//     if(pixelIndex < K){
//         if(counts[pixelIndex] > 0){
//             centroids_W[pixelIndex].r = centroids_W[pixelIndex].r/counts[pixelIndex];
//             centroids_W[pixelIndex].g = centroids_W[pixelIndex].g/counts[pixelIndex];
//             centroids_W[pixelIndex].b = centroids_W[pixelIndex].b/counts[pixelIndex];
//         }
//     }

//     __syncthreads();

//     int clusterId = clusterIds[thread_num];
//     d_outputImage[pixelIndex] = centroids_W[clusterId];

// }


// // =================================================================================
// // Host Code
// // =================================================================================

// int main() {
//     const auto init_start = std::chrono::steady_clock::now();
//     int K = 8;
//     int MAX_ITERATIONS = 20;

//     int IMG_WIDTH = 0;
//     int IMG_HEIGHT = 0;

//     std::cout << "Starting K-Means Color Clustering..." << std::endl;
//     std::cout << "  Clusters (K): " << K << std::endl;
//     std::cout << "  Max Iterations: " << MAX_ITERATIONS << std::endl;
//     std::cout << "------------------------------------" << std::endl;

//     std::vector<Point> h_input_points;
//     std::string inputFilename = "../img/camera_man.ppm"; 

//     std::ifstream ppm_file(inputFilename, std::ios::in | std::ios::binary);
//     if (!ppm_file) {
//         std::cerr << "Error: Could not open file '" << inputFilename << "'. Please check the path." << std::endl;
//         return 1;
//     }

//     std::string line;
//     ppm_file >> line; // Read "P6"
//     while (ppm_file.peek() == '\n' || ppm_file.peek() == '#') { ppm_file.ignore(256, '\n'); }
//     ppm_file >> IMG_WIDTH >> IMG_HEIGHT;
//     ppm_file.ignore(256, '\n'); // Skip max value line
//     ppm_file.ignore(256, '\n');

//     std::cout << "Reading image '" << inputFilename << "' (" << IMG_WIDTH << "x" << IMG_HEIGHT << ")" << std::endl;
    
//     std::vector<unsigned char> raw_pixel_data(IMG_WIDTH * IMG_HEIGHT * 3);
//     ppm_file.read(reinterpret_cast<char*>(raw_pixel_data.data()), raw_pixel_data.size());
//     ppm_file.close();

//     h_input_points.resize(IMG_WIDTH * IMG_HEIGHT);
//     for (size_t i = 0; i < h_input_points.size(); ++i) {
//         h_input_points[i] = {(float)raw_pixel_data[i*3], (float)raw_pixel_data[i*3+1], (float)raw_pixel_data[i*3+2]};
//     }
//     std::cout << "Loaded " << h_input_points.size() << " pixels as data points." << std::endl;

//     int numPoints = IMG_WIDTH * IMG_HEIGHT;
//     int imageBytes = numPoints * sizeof(Point);

//     // Error 4: Declared each pointer on its own or with a star
//     Point* d_inputImage;
//     Point* d_outputImage;
//     Point* d_centroids;

//     cudaMalloc(&d_inputImage, imageBytes);
//     cudaMalloc(&d_outputImage, imageBytes);
//     cudaMalloc(&d_centroids, K * sizeof(Point));

//     // Error 5: Used .data() to get the pointer from the vector
//     cudaMemcpy(d_inputImage, h_input_points.data(), imageBytes, cudaMemcpyHostToDevice);


//     // Initialize centroids by picking random pixels
//     std::vector<Point> h_centroids(K);
//     std::mt19937 rng(static_cast<unsigned int>(time(0)));
//     std::uniform_int_distribution<int> dist(0, numPoints - 1);
    
//     for (int i = 0; i < K; i++) {
//         h_centroids[i] = h_input_points[dist(rng)];
//     }
//     cudaMemcpy(d_centroids, h_centroids.data(), K * sizeof(Point), cudaMemcpyHostToDevice);

//     // Setup Grid and Block dimensions
//     int blockSize = 256;
//     int gridSize = (numPoints + blockSize - 1) / blockSize;
//     dim3 blockDim(blockSize);
//     dim3 gridDim(gridSize);
    
//     dim3 centroidGridDim((K + 255) / 256);
//     dim3 centroidBlockDim(256);

//     const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
//     std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

//     const auto compute_start = std::chrono::steady_clock::now();


//     std::cout << "Running K-Means algorithm on GPU..." << std::endl;
//     for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
//         k_mean_v2<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, d_centroids, numPoints, K);
//         cudaDeviceSynchronize();
//     }
//     std::cout << "K-Means iterations complete." << std::endl;

//     std::cout << "------------------------------------" << std::endl;
//     std::cout << "Generating quantized image data..." << std::endl;
    
//     std::vector<Point> h_output_points(numPoints);
//     cudaMemcpy(h_output_points.data(), d_outputImage, imageBytes, cudaMemcpyDeviceToHost);
    
//     const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
//     std::cout << "Computation time (sec): " << compute_time << '\n';


//     std::vector<unsigned char> result_image(numPoints * 3);
//     for (size_t i = 0; i < h_output_points.size(); ++i) {
//         result_image[i * 3 + 0] = static_cast<unsigned char>(h_output_points[i].r);
//         result_image[i * 3 + 1] = static_cast<unsigned char>(h_output_points[i].g);
//         result_image[i * 3 + 2] = static_cast<unsigned char>(h_output_points[i].b);
//     }
//     std::cout << "Image data stored in vector." << std::endl;
    
//     save_image_to_ppm("kmeans_quantized.ppm", result_image, IMG_WIDTH, IMG_HEIGHT);

//     cudaFree(d_inputImage);
//     cudaFree(d_outputImage);
//     cudaFree(d_centroids);

//     return 0;
// }


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

struct Point {
    float r, g, b;
};

// =================================================================================
// Helper Function (for saving the output)
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
// CUDA Kernel (Device Code)
// =================================================================================

/**
 * A device helper function to calculate the squared distance between two colors.
 */
__device__ float color_distance_sq(Point p1, Point p2) {
    float dr = p1.r - p2.r;
    float dg = p1.g - p2.g;
    float db = p1.b - p2.b;
    return dr*dr + dg*dg + db*db;
}

// =================================================================================
// KERNEL EDITED AS REQUESTED
// NOTE: This kernel still contains a fundamental design flaw. K-Means requires a global
// synchronization between the assignment step and the update step. A single kernel launch
// cannot perform a global sync. The results calculated here are only based on data
// within a single block and are NEVER written back to global memory to inform the next
// iteration. The algorithm will not converge correctly.
// =================================================================================
__global__ void k_means_v2(Point* d_inputImage, Point* d_outputImage, 
                            Point* d_centroids, int* d_counts, int numPoints, int K){

    // FIX 1: Corrected thread index calculation for a 1D grid.
    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixelIndex >= numPoints){
        return;
    }
    int thread_num = threadIdx.x;

    // Use shared memory.
    __shared__ Point centroids_W[K];
    __shared__ int counts[K];
    __shared__ int clusterIds[blockDim.x];
    __shared__ Point data_point[blockDim.x];


    // block's private shared memory.
    if(thread_num < K){
        centroids_W[thread_num] = d_centroids[thread_num];
        counts[thread_num] = 0; 
    }
    __syncthreads(); // Wait for all threads in the block to see the initialized centroids.


    // --- ASSIGNMENT STEP ---
    // This part is logically correct. Each thread finds the closest centroid
    // from the block's local copy.
    Point pixelColor = d_inputImage[pixelIndex];
    float min_dist = 1e30f; 
    int best_centroid = 0;

    for(int i = 0; i < K; i++){
        float dist = color_distance_sq(pixelColor, centroids_W[i]);
        if(min_dist > dist){
            min_dist = dist;
            best_centroid = i;
        }
    }
    
    // Store results for this thread in shared memory for the block-local reduction.
    data_point[thread_num] = pixelColor;
    clusterIds[thread_num] = best_centroid;

    __syncthreads();

    // --- UPDATE STEP (FUNDAMENTALLY FLAWED) ---
    // WARNING: This reduction only operates on the pixels processed by THIS block.
    // It calculates new centroids based on a tiny fraction of the image and ignores all other blocks.
    if(thread_num == 0){
        Point tmp_datapoints[32] = {0};
    
        for(int j = 0; j < blockDim.x; j++){
            if (blockIdx.x * blockDim.x + j < numPoints) { // Boundary check
                int clusterId = clusterIds[j];
                tmp_datapoints[clusterId].r += data_point[j].r;
                tmp_datapoints[clusterId].g += data_point[j].g;
                tmp_datapoints[clusterId].b += data_point[j].b;
                counts[clusterId] += 1;
            }
        }

        // Atomically update this block's shared `centroids_W` and `counts`.
        // This is unnecessary since only thread 0 is doing it, but we keep it.
        for(int z = 0; z < K; z++){
            atomicAdd(&d_centroids[z].r, tmp_datapoints[z].r);
            atomicAdd(&d_centroids[z].g, tmp_datapoints[z].g);
            atomicAdd(&d_centroids[z].b, tmp_datapoints[z].b);
            atomicAdd(&d_counts[z], counts[z]);
        }
    }   

    __syncthreads();

    // WARNING: Calculating the mean here is also flawed. It is only the mean
    // for the pixels within this block.
    if(pixelIndex < K){
        int count_num = d_counts[pixelIndex];
        if(count_num > 0){
            d_centroids[pixelIndex].r = d_centroids[pixelIndex].r / count_num;
            d_centroids[pixelIndex].g = d_centroids[pixelIndex].g / count_num;
            d_centroids[pixelIndex].b = d_centroids[pixelIndex].b / count_num;
        }
    }
    
    // WARNING: CRITICAL FLAW. The new centroids computed in `centroids_W` are
    // NEVER written back to `d_centroids` in global memory. This means the next
    // iteration of the k-means loop in `main` will use the same old centroids.
    // The algorithm DOES NOT PROGRESS.

    __syncthreads();

    // --- OUTPUT STEP ---
    // The output image is colored based on the initial centroids passed to the kernel.
    // Because the centroids are never updated globally, the output image will be
    // the same after 1 iteration as it is after 20 iterations.
    int finalClusterId = clusterIds[thread_num];
    d_outputImage[pixelIndex] = d_centroids[finalClusterId];
}


// =================================================================================
// Host Code (Unchanged)
// =================================================================================

int main() {
    const auto init_start = std::chrono::steady_clock::now();
    int K = 8;
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
    ppm_file >> line; // Read "P6"
    while (ppm_file.peek() == '\n' || ppm_file.peek() == '#') { ppm_file.ignore(256, '\n'); }
    ppm_file >> IMG_WIDTH >> IMG_HEIGHT;
    ppm_file.ignore(256, '\n'); // Skip max value line
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
    int* d_counts;

    cudaMalloc(&d_inputImage, imageBytes);
    cudaMalloc(&d_outputImage, imageBytes);
    cudaMalloc(&d_centroids, K * sizeof(Point));
    cudaMalloc(&d_counts, K * sizeof(int));

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
    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);
    
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();


    std::cout << "Running K-Means algorithm on GPU..." << std::endl;
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        k_means_v2<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, d_centroids, d_counts, numPoints, K);
        cudaDeviceSynchronize();
    }
    std::cout << "K-Means iterations complete." << std::endl;

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
    
    save_image_to_ppm("kmeans_quantized.ppm", result_image, IMG_WIDTH, IMG_HEIGHT);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_centroids);
    cudaFree(d_counts);

    return 0;
}
