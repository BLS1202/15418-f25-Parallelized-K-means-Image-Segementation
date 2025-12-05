#include <iostream>
#include <cuda_runtime.h>
#include "HPCRenderer.h"
#include "../Image.h"
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <ctime>
#include <limits>


// =================================================================================
// CUDA Kernel (Device Code)
// =================================================================================

// --- AFTER (This correctly flips the image during the copy) ---
__global__ void kernel_copy_image(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    // Index for reading from the source `in` buffer
    int source_index = y * width + x;
    // Calculate the flipped y-coordinate for the destination
    int flipped_y = height - 1 - y;
    // Index for writing to the destination `out` buffer
    int dest_index = flipped_y * width + x;
    // Perform the copy from the source location to the flipped destination
    out[4 * dest_index + 0] = in[4 * source_index + 0];
    out[4 * dest_index + 1] = in[4 * source_index + 1];
    out[4 * dest_index + 2] = in[4 * source_index + 2];
    out[4 * dest_index + 3] = in[4 * source_index + 3];
}

/**
 * A device helper function to calculate the squared distance between two colors.
 */
__device__ float color_distance_sq(float4 p1, float4 p2) {
    // 1. Calculate the difference for each color channel (r, g, b).
    // 2. Square each of those differences.
    // 3. Return the sum of the squared differences.
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    float dw = p1.w - p2.w; // NOTE: Typically not used for color, but included for completeness.
    return dx*dx + dy*dy + dz*dz + dw*dw; // FIX: Usually distance is just in RGB space, not alpha.
}
/**
 * Kernel 1: Assigns each pixel to its nearest centroid.
 */
// FIX: Added 'width' as a parameter because kernels cannot access class members like 'm_width'.
__global__ void assign_clusters_kernel(const float4* d_inputImage, int* d_clusterIds, const float4* d_centroids, int numPoints, int k, int width) {
    // 1. Get the unique global ID for the current thread.
    // 2. Ensure the thread ID is within the bounds of the pixel array.
    // 3. Get the color of the pixel this thread is responsible for.
    // 4. Initialize variables to track the closest centroid found so far.
    // 5. Loop through all K centroids to find the closest one.
    //    a. Get the color of the current centroid.
    //    b. Calculate the distance between the pixel and the centroid.
    //    c. If this centroid is closer, update the minimum distance and the assigned cluster ID.
    // 6. Store the ID of the closest cluster in the output array.

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelIndex = y * width + x; // FIX: Used 'width' parameter instead of 'm_width'. Added missing semicolon.
    if(pixelIndex >= numPoints){
        return;
    }
    //get color
    float4 pixelColor = d_inputImage[pixelIndex]; // FIX: This variable was correct.
    
    //check centroid
    int best_centroid = 0; // FIX: Declared and initialized 'best_centroid'.
    float min_dist = 1e10f; // FIX: Initialize min_dist to a very large value.
    float new_dist; // FIX: Declared 'new_dist' before the loop.

    for(int i = 0; i < k; i++){ // FIX: Changed comma to semicolon in for loop condition.
        new_dist = color_distance_sq(pixelColor, d_centroids[i]);
        if(min_dist > new_dist){
            min_dist = new_dist;
            best_centroid = i;
        }
    }
    d_clusterIds[pixelIndex] = best_centroid;
}
/**
 * Kernel 2: Sums the colors and counts for each cluster using atomic operations.
 */
// FIX: Added 'width' as a parameter. Changed 'd_points' to 'd_inputImage' to match usage.
__global__ void update_centroids_kernel(const float4* d_inputImage, const int* d_clusterIds, float4* d_sums, int* d_counts, int numPoints, int width) {
    // 1. Get the unique global ID for the current thread.
    // 2. Ensure the thread ID is within the bounds of the pixel array.
    // 3. Find out which cluster this pixel was assigned to.
    // 4. Get the color of this pixel.
    // 5. Atomically add this pixel's color to the correct cluster's running total.
    // 6. Atomically increment the pixel counter for that cluster.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int pixelIndex = y * width + x; // FIX: Used 'width' parameter instead of 'm_width'.
    if(pixelIndex >= numPoints){
        return;
    }

    int clusterId = d_clusterIds[pixelIndex];
    //get color
    float4 pixelColor = d_inputImage[pixelIndex];
    atomicAdd(&(d_sums[clusterId].x), pixelColor.x);
    atomicAdd(&(d_sums[clusterId].y), pixelColor.y);
    atomicAdd(&(d_sums[clusterId].z), pixelColor.z);
    atomicAdd(&(d_sums[clusterId].w), pixelColor.w);
    atomicAdd(&(d_counts[clusterId]), 1);
}
/**
 * Kernel 3: Calculates the new average color for each centroid.
 */
__global__ void calculate_new_centroids_kernel(float4* d_centroids, const float4* d_sums, const int* d_counts, int k) {
    // 1. Get the thread ID, which corresponds to the centroid index.
    // 2. Ensure the thread ID is within the bounds of the centroid array (0 to K-1).
    // 3. Get the total number of pixels assigned to this centroid.
    // 4. If the count is greater than zero:
    //    a. Get the total summed color for this cluster.
    //    b. Calculate the new average color by dividing the sum by the count.
    //    c. Update the centroid's color in the main centroid array.

    // FIX: This kernel iterates over K centroids, not pixels. The index is the 1D thread ID.
    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixelIndex >= k){
        return;
    }
    
    int count = d_counts[pixelIndex]; // FIX: This variable was correct.
    if(count > 0){
        float4 sum = d_sums[pixelIndex];
        d_centroids[pixelIndex].x = sum.x/count;
        d_centroids[pixelIndex].y = sum.y/count;
        d_centroids[pixelIndex].z = sum.z/count;
        d_centroids[pixelIndex].w = sum.w/count;
    }
    
}
/**
 * Kernel 4: Generates the final output image from the clustering results.
 */
// FIX: Added 'width' as a parameter.
__global__ void generate_output_image_kernel(float4* d_outputImage, const int* d_clusterIds, const float4* d_centroids, int numPoints, int width, int height) {
    // 1. Get the unique global ID for the current thread.
    // 2. Ensure the thread ID is within the bounds of the pixel array.
    // 3. Find out which cluster this pixel belongs to.
    // 4. Get the final color of that cluster's centroid.
    // 5. Write this color to the corresponding pixel in the output image buffer.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelIndex = y * width + x; // FIX: Used 'width' parameter instead of 'm_width'. Added missing semicolon.
    if(pixelIndex >= numPoints){
        return;
    }
    int clusterId = d_clusterIds[pixelIndex]; // FIX: This variable was correct.
    float4 newColor = d_centroids[clusterId];

    int flipped_y = height - 1 - y;
    int output_index = flipped_y * width + x;

    d_outputImage[output_index] = newColor;

}



// =================================================================================
// C++ Class Implementation (Host Code)
// =================================================================================

HPCRenderer::HPCRenderer(Image* image){

    m_k = 3;
    m_maxIterations = 50; // Increased iterations for better convergence
    m_currentIteration = 0;

    m_width = image->width;
    m_height = image->height;
    m_numPoints = m_width * m_height;

    size_t imageBytes = sizeof(float4) * m_width * m_height;

    cudaMalloc(&d_centroids, m_k*sizeof(float4));
    cudaMalloc(&d_clusterIds, m_numPoints*sizeof(int));
    cudaMalloc(&d_inputImage, imageBytes);
    cudaMalloc(&d_outputImage, imageBytes);
    cudaMalloc(&d_sums, m_k*sizeof(float4));
    cudaMalloc(&d_counts, m_k*sizeof(int));


    cudaMemcpy(d_inputImage, image->data, imageBytes, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((m_width + blockDim.x - 1) / blockDim.x, 
                 (m_height + blockDim.y - 1) / blockDim.y);
    kernel_copy_image<<<gridDim, blockDim>>>(
        (float*)d_inputImage, 
        (float*)d_outputImage, 
        m_width, 
        m_height
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after setup: %s\n", cudaGetErrorString(err));
    }
}

HPCRenderer::~HPCRenderer(){
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_centroids);
    cudaFree(d_clusterIds);
    cudaFree(d_sums);
    cudaFree(d_counts);
}

Image* HPCRenderer::getDisplayImage() {
    Image* h_image = new Image(m_width, m_height);
    size_t imageBytes = sizeof(float4) * m_width * m_height;
    
    // Copy the contents of d_outputImage (device) to h_image->data (host).
    cudaMemcpy(h_image->data, d_outputImage, imageBytes, cudaMemcpyDeviceToHost);

    return h_image;
}

void HPCRenderer::startKMeansSegmentation() {
    std::cout << "Initializing K-Means segmentation..." << std::endl;
    
    m_currentIteration = 0; 

    // Create a host-side vector to hold the original image data
    std::vector<float4> h_inputImage(m_numPoints);
    cudaMemcpy(h_inputImage.data(), d_inputImage, sizeof(float4) * m_numPoints, cudaMemcpyDeviceToHost);

    std::vector<float4> h_centroids;
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, m_numPoints - 1);
    
    // Populate the vector by randomly picking K pixels from the host copy of the image.
    for(int i = 0; i < m_k; i++){
        h_centroids.push_back(h_inputImage[dist(rng)]);
    }
    
    cudaMemcpy(d_centroids, h_centroids.data(), sizeof(float4)*m_k, cudaMemcpyHostToDevice);
}

void HPCRenderer::stepKMeansIteration() {
    if (isKMeansDone()) {
        return;
    }
    
    std::cout << "Running K-Means Iteration: " << m_currentIteration + 1 << std::endl;

    dim3 blockDim(16, 16);
    dim3 gridDim((m_width + blockDim.x - 1) / blockDim.x, 
                 (m_height + blockDim.y - 1) / blockDim.y);
    
    // Kernel for centroids (1D)
    dim3 centroidGridDim((m_k + 255) / 256);
    dim3 centroidBlockDim(256);

    assign_clusters_kernel<<<gridDim, blockDim>>>(d_inputImage, d_clusterIds, d_centroids, m_numPoints, m_k, m_width);
    
    cudaMemset(d_sums, 0, sizeof(float4)*m_k);
    cudaMemset(d_counts, 0, sizeof(int)*m_k);
    cudaDeviceSynchronize(); // Good practice to check for errors after kernels
    

    update_centroids_kernel<<<gridDim, blockDim>>>(d_inputImage, d_clusterIds, d_sums, d_counts, m_numPoints, m_width);
    cudaDeviceSynchronize();
    

    calculate_new_centroids_kernel<<<centroidGridDim, centroidBlockDim>>>(d_centroids, d_sums, d_counts, m_k);
    cudaDeviceSynchronize();
    

    generate_output_image_kernel<<<gridDim, blockDim>>>(d_outputImage, d_clusterIds, d_centroids, m_numPoints, m_width, m_height);
    cudaDeviceSynchronize();
    
    m_currentIteration++;
}
bool HPCRenderer::isKMeansDone() const {
    return m_currentIteration >= m_maxIterations;
}
