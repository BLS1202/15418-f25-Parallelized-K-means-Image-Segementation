#include <iostream>
#include <cuda_runtime.h>
#include "OpenMPrenderer.h"
#include "../Image.h"
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <ctime>
#include <limits>
#include <omp.h> // Include OpenMP header


// =================================================================================
// K-Means Helper Structures & Functions (from k_mean_image_openmp2.cpp)
// =================================================================================

struct Point {
    double r, g, b;
    int clusterId;
};

// 3D Euclidean distance between colors
inline double distance(Point p1, Point p2) {
    return std::sqrt(std::pow(p1.r - p2.r, 2) + std::pow(p1.g - p2.g, 2) + std::pow(p1.b - p2.b, 2));
}

// =================================================================================
// CUDA Kernel (Device Code)
// =================================================================================


__global__ void kernel_copy_image2(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int source_index = y * width + x;

    int flipped_y = height - 1 - y;

    int dest_index = flipped_y * width + x;

    out[4 * dest_index + 0] = in[4 * source_index + 0];
    out[4 * dest_index + 1] = in[4 * source_index + 1];
    out[4 * dest_index + 2] = in[4 * source_index + 2];
    out[4 * dest_index + 3] = in[4 * source_index + 3];
}


__forceinline__ __device__ float color_distance_sq(float4 p1, float4 p2) {

    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    float dw = p1.w - p2.w; 
    return dx*dx + dy*dy + dz*dz + dw*dw; 
}

// =================================================================================
// C++ Class Implementation (Host Code)
// =================================================================================

OpenMPRenderer::OpenMPRenderer(Image* image): Renderer(image){

    m_k = 8;
    m_maxIterations = 20;
    m_currentIteration = 0;
    m_converged = false;

    m_width = image->width;
    m_height = image->height;
    m_numPoints = m_width * m_height;

    size_t imageBytes = sizeof(float4) * m_numPoints;

    cudaMalloc(&d_inputImage, imageBytes);
    cudaMalloc(&d_outputImage, imageBytes);

    cudaMemcpy(d_inputImage, image->data, imageBytes, cudaMemcpyHostToDevice);


    dim3 blockDim(16, 16);
    dim3 gridDim((m_width + blockDim.x - 1) / blockDim.x, 
                 (m_height + blockDim.y - 1) / blockDim.y);
    kernel_copy_image2<<<gridDim, blockDim>>>(
        (float*)d_inputImage, 
        (float*)d_outputImage, 
        m_width, 
        m_height
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after setup: %s\n", cudaGetErrorString(err));
    }


    std::vector<float4> h_inputImage(m_numPoints);
    cudaMemcpy(h_inputImage.data(), d_inputImage, imageBytes, cudaMemcpyDeviceToHost);

    m_points.resize(m_numPoints);
    for (int i = 0; i < m_numPoints; ++i) {
        m_points[i] = {
            (double)h_inputImage[i].x * 255.0,
            (double)h_inputImage[i].y * 255.0,
            (double)h_inputImage[i].z * 255.0,
            -1 // initial clusterId
        };
    }
}

OpenMPRenderer::~OpenMPRenderer(){
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

Image* OpenMPRenderer::getDisplayImage() {
    Image* h_image = new Image(m_width, m_height);
    size_t imageBytes = sizeof(float4) * m_width * m_height;
    
    // Copy the contents of d_outputImage (device) to h_image->data (host).
    cudaMemcpy(h_image->data, d_outputImage, imageBytes, cudaMemcpyDeviceToHost);

    return h_image;
}

void OpenMPRenderer::startKMeansSegmentation() {
    std::cout << "Initializing K-Means segmentation with OpenMP..." << std::endl;
    
    m_currentIteration = 0; 
    m_converged = false;

    // Initialize Centroids randomly from the points
    m_centroids.clear();
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, m_points.size() - 1);
    for (int i = 0; i < m_k; ++i) {
        m_centroids.push_back(m_points[dist(rng)]);
    }

    std::cout << "K-Means initialized. Ready to start iterations." << std::endl;
}

void OpenMPRenderer::stepKMeansIteration() {
    if (isKMeansDone()) {
        return;
    }
    
    // --- K-Means Iteration Logic (from k_mean_image_openmp2.cpp) ---

    int changed = 0;
    // Assignment Step
    #pragma omp parallel for reduction(||:changed)
    for (int p = 0; p < m_points.size(); p++) {
        double min_dist = std::numeric_limits<double>::max();
        int closest_centroid_id = -1;
        for (int i = 0; i < m_k; ++i) {
            double d = distance(m_points[p], m_centroids[i]);
            if (d < min_dist) {
                min_dist = d; 
                closest_centroid_id = i;
            }
        }
        if(m_points[p].clusterId != closest_centroid_id){
            m_points[p].clusterId = closest_centroid_id;
            changed = 1; // some pixels changed its cluster
        }
    }
    
    // Update Step
    std::vector<Point> new_centroids(m_k, {0, 0, 0, -1});
    std::vector<double> r(m_k, 0.0), g(m_k, 0.0), b(m_k, 0.0); // Use double for sums
    std::vector<int> counts(m_k, 0);

    double r_arr[m_k] = {0};
    double g_arr[m_k] = {0};
    double b_arr[m_k] = {0};
    int counts_arr[m_k] = {0};

    #pragma omp parallel for reduction(+:r_arr[:m_k], g_arr[:m_k], b_arr[:m_k], counts_arr[:m_k])
    for (int p = 0; p < m_points.size(); p++){
        int cluster_id = m_points[p].clusterId;
        if (cluster_id != -1) {
            r_arr[cluster_id] += m_points[p].r;
            g_arr[cluster_id] += m_points[p].g; 
            b_arr[cluster_id] += m_points[p].b; 
            counts_arr[cluster_id] += 1;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < m_k; i++) {
        if (counts_arr[i] > 0) {
            m_centroids[i].r = r_arr[i] / counts_arr[i]; 
            m_centroids[i].g = g_arr[i] / counts_arr[i]; 
            m_centroids[i].b = b_arr[i] / counts_arr[i]; 
        }
    }

    // Check for convergence
    if (changed == 0) {
        m_converged = true;
        std::cout << "K-Means converged after " << m_currentIteration << " iterations." << std::endl;
    }

    // --- Update the output image on the GPU for display ---
    std::vector<float4> h_resultImage(m_numPoints);
    #pragma omp parallel for
    for (int i = 0; i < m_points.size(); i++) {
        Point centroid_color = m_centroids[m_points[i].clusterId];
        // Convert back from double [0, 255] to float [0, 1]
        h_resultImage[i].x = static_cast<float>(centroid_color.r / 255.0);
        h_resultImage[i].y = static_cast<float>(centroid_color.g / 255.0);
        h_resultImage[i].z = static_cast<float>(centroid_color.b / 255.0);
        h_resultImage[i].w = 1.0f; // Alpha
    }

    // Copy the updated host image to the d_inputImage buffer (as source for kernel)
    cudaMemcpy(d_inputImage, h_resultImage.data(), sizeof(float4) * m_numPoints, cudaMemcpyHostToDevice);
    
    // Use the kernel to copy and flip the image to the final display buffer
    dim3 blockDim(16, 16);
    dim3 gridDim((m_width + blockDim.x - 1) / blockDim.x, 
                 (m_height + blockDim.y - 1) / blockDim.y);
    kernel_copy_image2<<<gridDim, blockDim>>>(
        (float*)d_inputImage, 
        (float*)d_outputImage, 
        m_width, 
        m_height
    );

    m_currentIteration++;
}

bool OpenMPRenderer::isKMeansDone() const {
    return m_currentIteration >= m_maxIterations || m_converged;
}

