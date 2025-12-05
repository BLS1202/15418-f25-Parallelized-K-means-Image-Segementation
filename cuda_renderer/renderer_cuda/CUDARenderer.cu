#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cuda_runtime.h>

#include "CUDARenderer.h"
#include "../Image.h"

// =============================================================================
// K-Means Configuration
// =============================================================================
#define THREAD_X 32
#define THREAD_Y 32
#define BLOCKSIZE (THREAD_X * THREAD_Y)

// =============================================================================
// CUDA Kernels adapted for float4 (Unchanged from before)
// =============================================================================

// Device helper function to calculate squared distance between two float4 colors (ignores alpha)
__forceinline__ __device__ float color_distance_sq(float4 p1, float4 p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return dx*dx + dy*dy + dz*dz;
}

__global__ void k_means_v2(float4* d_inputImage, float4* d_centroid_sums, 
    int* d_counts, float4* d_centroids, int numPoints, int width, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelIndex = y * width + x;
    int thread_num = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float4 data_point[BLOCKSIZE];
    __shared__ float4 partial_sums[256];
    __shared__ int partial_counts[256];
    __shared__ float4 centroids_W[256];

    if (pixelIndex < numPoints) {
        data_point[thread_num] = d_inputImage[pixelIndex];
    }

    if (thread_num < k) {
        partial_sums[thread_num] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        partial_counts[thread_num] = 0;
        centroids_W[thread_num] = d_centroids[thread_num];
    }
    __syncthreads();

    if (pixelIndex < numPoints) {
        float4 pixelColor = data_point[thread_num];
        float min_dist = 1e30f;
        int best_centroid_id = 0;

        for (int i = 0; i < k; i++) {
            float dist = color_distance_sq(pixelColor, centroids_W[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_centroid_id = i;
            }
        }

        atomicAdd(&partial_sums[best_centroid_id].x, data_point[thread_num].x);
        atomicAdd(&partial_sums[best_centroid_id].y, data_point[thread_num].y);
        atomicAdd(&partial_sums[best_centroid_id].z, data_point[thread_num].z);
        atomicAdd(&partial_counts[best_centroid_id], 1);
    }
    __syncthreads();

    if (thread_num == 0) {
        for (int i = 0; i < k; i++) {
            if (partial_counts[i] > 0) {
                atomicAdd(&d_centroid_sums[i].x, partial_sums[i].x);
                atomicAdd(&d_centroid_sums[i].y, partial_sums[i].y);
                atomicAdd(&d_centroid_sums[i].z, partial_sums[i].z);
                atomicAdd(&d_counts[i], partial_counts[i]);
            }
        }
    }
}

// Kernel 2: Calculate new centroid averages, adapted for float4
__global__ void update_centroids(float4* d_centroids, float4* d_centroid_sums, int* d_counts, int k) {
    int i = threadIdx.x;
    if (i >= k) return;
    if (d_counts[i] > 0) {
        d_centroids[i].x = d_centroid_sums[i].x / d_counts[i];
        d_centroids[i].y = d_centroid_sums[i].y / d_counts[i];
        d_centroids[i].z = d_centroid_sums[i].z / d_counts[i];
        d_centroids[i].w = 255.0f;
    }
}

// Kernel 3: Generate output image, adapted for float4
__global__ void generate_output_image(float4* d_outputImage, float4* d_inputImage, float4* d_centroids, int numPoints, int width, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_num = threadIdx.y * blockDim.x + threadIdx.x;
    int pixelIndex = y * width + x;

    __shared__ float4 centroids_W[256];

    if (pixelIndex >= numPoints) return;

    if (thread_num < k) {
        centroids_W[thread_num] = d_centroids[thread_num];
    }
    __syncthreads();
    
    float4 pixelColor = d_inputImage[pixelIndex];
    float min_dist = 1e30f;
    int best_centroid_id = 0;

    for (int i = 0; i < k; i++) {
        float dist = color_distance_sq(pixelColor, centroids_W[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid_id = i;
        }
    }
    d_outputImage[pixelIndex] = centroids_W[best_centroid_id];
}

// =============================================================================
// CUDARenderer Class Implementation
// =============================================================================

CUDARenderer::CUDARenderer(Image* image) : Renderer(image){
    m_width = image->width;
    m_height = image->height;
    m_numPoints = m_width * m_height;
    
    m_k = 8;
    m_maxIterations = 20;
    m_currentIteration = 0;

    std::cout << "Initializing CUDARenderer with float4 data structure." << std::endl;

    h_input_points.resize(m_numPoints);
    for (int i = 0; i < m_numPoints; ++i) {
        h_input_points[i].x = image->data[4 * i + 0];
        h_input_points[i].y = image->data[4 * i + 1]; 
        h_input_points[i].z = image->data[4 * i + 2]; 
        h_input_points[i].w = image->data[4 * i + 3]; 
    }

    size_t imageBytes = m_numPoints * sizeof(float4);
    cudaMalloc(&d_inputImage, imageBytes);
    cudaMalloc(&d_outputImage, imageBytes);
    cudaMalloc(&d_centroids, m_k * sizeof(float4));
    cudaMalloc(&d_centroid_sums, m_k * sizeof(float4));
    cudaMalloc(&d_counts, m_k * sizeof(int));

    cudaMemcpy(d_inputImage, h_input_points.data(), imageBytes, cudaMemcpyHostToDevice);
}

CUDARenderer::~CUDARenderer() {
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_centroids);
    cudaFree(d_centroid_sums);
    cudaFree(d_counts);
}

void CUDARenderer::startKMeansSegmentation() {
    m_currentIteration = 0; 
    std::cout << "Starting K-Means: Initializing random centroids..." << std::endl;

    std::vector<float4> h_centroids(m_k);
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, m_numPoints - 1);

    for (int i = 0; i < m_k; i++) {
        h_centroids[i] = h_input_points[dist(rng)];
    }
    cudaMemcpy(d_centroids, h_centroids.data(), m_k * sizeof(float4), cudaMemcpyHostToDevice);
}

void CUDARenderer::stepKMeansIteration() {
    if (isKMeansDone()) return;
    
    m_currentIteration++;
    std::cout << "Running K-Means Iteration: " << m_currentIteration << "/" << m_maxIterations << std::endl;

    dim3 blockDim(THREAD_X, THREAD_Y);
    dim3 gridDim((m_width + THREAD_X - 1) / THREAD_X, (m_height + THREAD_Y - 1) / THREAD_Y);

    cudaMemset(d_centroid_sums, 0, m_k * sizeof(float4));
    cudaMemset(d_counts, 0, m_k * sizeof(int));

    k_means_v2<<<gridDim, blockDim>>>(d_inputImage, d_centroid_sums, d_counts, d_centroids, m_numPoints, m_width, m_k);
    
    update_centroids<<<1, m_k>>>(d_centroids, d_centroid_sums, d_counts, m_k);
}

bool CUDARenderer::isKMeansDone() const {
    return m_currentIteration >= m_maxIterations;
}

Image* CUDARenderer::getDisplayImage() {
    std::cout << "Generating final display image..." << std::endl;

    dim3 blockDim(THREAD_X, THREAD_Y);
    dim3 gridDim((m_width + THREAD_X - 1) / THREAD_X, (m_height + THREAD_Y - 1) / THREAD_Y);
    generate_output_image<<<gridDim, blockDim>>>(d_outputImage, d_inputImage, d_centroids, m_numPoints, m_width, m_k);
    cudaDeviceSynchronize();

    Image* displayImage = new Image(m_width, m_height);
    std::vector<float4> h_output_points(m_numPoints);

    cudaMemcpy(h_output_points.data(), d_outputImage, m_numPoints * sizeof(float4), cudaMemcpyDeviceToHost);

    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            int source_index = y * m_width + x;
            int dest_y = m_height - 1 - y;
            int dest_index = dest_y * m_width + x;
            
            float4 color = h_output_points[source_index];
            displayImage->data[4 * dest_index + 0] = color.x; 
            displayImage->data[4 * dest_index + 1] = color.y; 
            displayImage->data[4 * dest_index + 2] = color.z; 
            displayImage->data[4 * dest_index + 3] = color.w;
        }
    }

    return displayImage;
}
