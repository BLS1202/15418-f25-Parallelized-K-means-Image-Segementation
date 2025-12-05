// ============================================================================
// CUDARenderer.cu — CUDA K-Means Renderer (global memory atomic version)
// ============================================================================
#include "CUDARenderer.h"
#include "../Image.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// --------------------------------------------------------------------------
// Device helper: squared distance between two colors
__device__ float color_distance_sq(float4 a, float4 b) {
    float dr = a.x - b.x;
    float dg = a.y - b.y;
    float db = a.z - b.z;
    return dr*dr + dg*dg + db*db;
}

__global__ void assign_and_compute_sums(
    const float4* d_input,
    int* d_clusterIds,
    float4* d_centroids,
    float4* d_centroid_sums,
    int* d_counts,
    int width,
    int height,
    int K
) {
    int tx = threadIdx.x + threadIdx.y * blockDim.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    int numPoints = width*height;
    
    __shared__ float4 data_point[BLOCK_DIM_X*BLOCK_DIM_Y];     
    __shared__ float4 partial_sums[8];          // length == K, K is 8 here
    __shared__ int partial_counts[8];           
    __shared__ float4 centroids_W[8];          

    if (x >= width || y >= height) return;
    // Load pixel data
    if (idx < numPoints) data_point[tx] = d_input[idx];

    // Initialize partial sums and counts
    if (tx < K) {
        partial_sums[tx] = make_float4(0,0,0,0);
        partial_counts[tx] = 0;
        centroids_W[tx] = d_centroids[tx];
    }

    __syncthreads();

    if (idx < numPoints) {
        float4 pix = data_point[tx];
        float minDist = 1e30f;
        int bestCluster = 0;

        for (int c = 0; c < K; c++) {
            float dr = pix.x - centroids_W[c].x;
            float dg = pix.y - centroids_W[c].y;
            float db = pix.z - centroids_W[c].z;
            float dist = dr*dr + dg*dg + db*db;
            if (dist < minDist) {
                minDist = dist;
                bestCluster = c;
            }
        }

        d_clusterIds[idx] = bestCluster;

        atomicAdd(&partial_sums[bestCluster].x, pix.x);
        atomicAdd(&partial_sums[bestCluster].y, pix.y);
        atomicAdd(&partial_sums[bestCluster].z, pix.z);
        atomicAdd(&partial_counts[bestCluster], 1);
    }

    __syncthreads();

    // Thread 0 updates global sums
    if (tx == 0) {
        for (int c = 0; c < K; c++) {
            if (partial_counts[c] > 0) {
                atomicAdd(&d_centroid_sums[c].x, partial_sums[c].x);
                atomicAdd(&d_centroid_sums[c].y, partial_sums[c].y);
                atomicAdd(&d_centroid_sums[c].z, partial_sums[c].z);
                atomicAdd(&d_counts[c], partial_counts[c]);
            }
        }
    }
}


// kernel update centroids
__global__ void update_centroids(float4* d_centroids, const float4* d_sums, const int* d_counts, int k) {
    int i = threadIdx.x;
    if (i >= k) return;
    int count = d_counts[i];
    if (count > 0) {
        float4 s = d_sums[i];
        d_centroids[i] = make_float4(s.x / count, s.y / count, s.z / count, 1.0f);
    }
}

// kernel generate output image
__global__ void generate_output_image(
    float4* d_output,
    const int* d_clusterIds,
    const float4* d_centroids,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;                     // original index
    int flippedY = height - 1 - y;              // flip vertically
    int outIdx = flippedY * width + x;          // flipped index

    int c = d_clusterIds[idx];
    float4 color = d_centroids[c];
    d_output[outIdx] = color;
}


// CudaRenderer Implementation
CudaRenderer::CudaRenderer(Image* image, int k)
    : Renderer(image)
{
    m_k = k;
    m_width = image->width;
    m_height = image->height;
    m_numPoints = m_width * m_height;
    m_currentIteration = 0;
    m_maxIterations = 20;

    m_blockDim = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);
    m_gridDim = dim3((m_width + BLOCK_DIM_X - 1)/BLOCK_DIM_X,
                     (m_height + BLOCK_DIM_Y - 1)/BLOCK_DIM_Y);

    size_t imgBytes = sizeof(float4) * m_numPoints;

    cudaMalloc(&d_inputImage, imgBytes);
    cudaMalloc(&d_outputImage, imgBytes);
    cudaMalloc(&d_centroids, sizeof(float4) * m_k);
    cudaMalloc(&d_centroid_sums, sizeof(float4) * m_k);
    cudaMalloc(&d_counts, sizeof(int) * m_k);
    cudaMalloc(&d_clusterIds, sizeof(int) * m_numPoints);

    // Copy host image to device
    std::vector<float4> h_input(m_numPoints);
    for (int i = 0; i < m_numPoints; ++i) {
        h_input[i] = make_float4(
            image->data[i*4 + 0],
            image->data[i*4 + 1],
            image->data[i*4 + 2],
            1.0f
        );
    }
    cudaMemcpy(d_inputImage, h_input.data(), imgBytes, cudaMemcpyHostToDevice);

    m_displayImage = new Image(m_width, m_height);
}

CudaRenderer::~CudaRenderer() {
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_centroids);
    cudaFree(d_centroid_sums);
    cudaFree(d_counts);
    cudaFree(d_clusterIds);
    delete m_displayImage;
}


void CudaRenderer::startKMeansSegmentation() {
    std::vector<float4> h_input(m_numPoints);
    cudaMemcpy(h_input.data(), d_inputImage, sizeof(float4)*m_numPoints, cudaMemcpyDeviceToHost);

    std::vector<float4> h_centroids(m_k);
    std::mt19937 rng(time(0));
    std::uniform_int_distribution<int> dist(0, m_numPoints-1);

    for (int i = 0; i < m_k; ++i)
        h_centroids[i] = h_input[dist(rng)];

    cudaMemcpy(d_centroids, h_centroids.data(), sizeof(float4)*m_k, cudaMemcpyHostToDevice);
    m_currentIteration = 0;
}


void CudaRenderer::stepKMeansIteration() {
    if (m_currentIteration >= m_maxIterations) return;

    cudaMemset(d_centroid_sums, 0, sizeof(float4)*m_k);
    cudaMemset(d_counts, 0, sizeof(int)*m_k);

    assign_and_compute_sums<<<m_gridDim, m_blockDim>>>(d_inputImage, d_clusterIds, d_centroids,
    d_centroid_sums,
    d_counts,
    m_width,
    m_height,
    m_k);
    cudaDeviceSynchronize();

    // 3. Update centroids
    update_centroids<<<1, m_k>>>(d_centroids, d_centroid_sums, d_counts, m_k);
    cudaDeviceSynchronize();

    // 4. Generate output image
    generate_output_image<<<m_gridDim, m_blockDim>>>(d_outputImage, d_clusterIds, d_centroids, m_width, m_height);
    cudaDeviceSynchronize();

    // Copy to host
    cudaMemcpy(m_displayImage->data, d_outputImage, sizeof(float4)*m_numPoints, cudaMemcpyDeviceToHost);

    m_currentIteration++;
}


bool CudaRenderer::isKMeansDone() const {
    return m_currentIteration >= m_maxIterations;
}

Image* CudaRenderer::getDisplayImage() {
    return m_displayImage;
}
