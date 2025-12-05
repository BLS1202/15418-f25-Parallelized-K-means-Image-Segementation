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




struct Point {
    double r, g, b;
    int clusterId;
};

// 3D Euclidean distance between colors
inline double distance(Point p1, Point p2) {
    return std::sqrt(std::pow(p1.r - p2.r, 2) + std::pow(p1.g - p2.g, 2) + std::pow(p1.b - p2.b, 2));
}

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



HPCRenderer::HPCRenderer(Image* image) : Renderer(image){

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

    // Copy original image to output for initial display
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

    // Prepare host data for sequential K-Means by copying from GPU
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

HPCRenderer::~HPCRenderer(){
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

Image* HPCRenderer::getDisplayImage() {
    Image* h_image = new Image(m_width, m_height);
    size_t imageBytes = sizeof(float4) * m_width * m_height;
    
    cudaMemcpy(h_image->data, d_outputImage, imageBytes, cudaMemcpyDeviceToHost);

    return h_image;
}

void HPCRenderer::startKMeansSegmentation() {
    std::cout << "Initializing K-Means segmentation..." << std::endl;
    
    m_currentIteration = 0;
    m_converged = false;


    m_centroids.clear();
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, m_points.size() - 1);
    for (int i = 0; i < m_k; ++i) {
        m_centroids.push_back(m_points[dist(rng)]);
    }
    
    std::cout << "K-Means initialized. Ready to start iterations." << std::endl;
}

void HPCRenderer::stepKMeansIteration() {
    if (isKMeansDone()) {
        return;
    }
    
    // --- K-Means Iteration Logic (from k_mean_image.cpp) ---
    bool changed = false;

    // Assignment Step
    for (auto& point : m_points) {
        double min_dist = std::numeric_limits<double>::max();
        int closest_centroid_id = -1;
        for (int i = 0; i < m_k; ++i) {
            double d = distance(point, m_centroids[i]);
            if (d < min_dist) { min_dist = d; closest_centroid_id = i; }
        }
        if (point.clusterId != closest_centroid_id) {
            point.clusterId = closest_centroid_id;
            changed = true;
        }
    }

    // Update Step
    std::vector<Point> new_centroids(m_k, {0, 0, 0, -1});
    std::vector<int> counts(m_k, 0);
    for (const auto& point : m_points) {
        int cluster_id = point.clusterId;
        if(cluster_id != -1) {
            new_centroids[cluster_id].r += point.r;
            new_centroids[cluster_id].g += point.g;
            new_centroids[cluster_id].b += point.b;
            counts[cluster_id]++;
        }
    }
    for (int i = 0; i < m_k; ++i) {
        if (counts[i] > 0) {
            m_centroids[i].r = new_centroids[i].r / counts[i];
            m_centroids[i].g = new_centroids[i].g / counts[i];
            m_centroids[i].b = new_centroids[i].b / counts[i];
        }
    }

    // Convergence Check
    if (!changed) {
        m_converged = true;
        std::cout << "K-Means converged after " << m_currentIteration << " iterations." << std::endl;
    }

    // --- Update the output image on the GPU for display ---
    std::vector<float4> h_resultImage(m_numPoints);
    for (size_t i = 0; i < m_points.size(); ++i) {
        Point centroid_color = m_centroids[m_points[i].clusterId];
        h_resultImage[i].x = static_cast<float>(centroid_color.r / 255.0);
        h_resultImage[i].y = static_cast<float>(centroid_color.g / 255.0);
        h_resultImage[i].z = static_cast<float>(centroid_color.b / 255.0);
        h_resultImage[i].w = 1.0f; // Alpha
    }


    cudaMemcpy(d_inputImage, h_resultImage.data(), sizeof(float4) * m_numPoints, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((m_width + blockDim.x - 1) / blockDim.x, 
                 (m_height + blockDim.y - 1) / blockDim.y);
    kernel_copy_image<<<gridDim, blockDim>>>(
        (float*)d_inputImage, 
        (float*)d_outputImage, 
        m_width, 
        m_height
    );

    m_currentIteration++;
}
bool HPCRenderer::isKMeansDone() const {
    return m_currentIteration >= m_maxIterations || m_converged;
}
