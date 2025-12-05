#pragma once
#include "../Renderer.h"
class Image;

struct float4;
class HPCRenderer : public Renderer {
public:
    // Constructor: Allocates all GPU memory for display and K-Means.
    HPCRenderer(Image* image);

    // Destructor: Frees all allocated GPU memory.
    virtual ~HPCRenderer();

    // Fetches the current rendered image from the GPU for display.
    // The caller is responsible for deleting the returned Image object.
    Image* getDisplayImage();

    // --- K-Means Control Functions ---

    // Resets the algorithm, picks K random centroids, and resets the iteration count.
    void startKMeansSegmentation();

    // Executes a single iteration of the K-Means algorithm.
    void stepKMeansIteration();

    bool isKMeansDone() const;
private:
    // --- Display & Image Properties ---
    int m_width;
    int m_height;
    int m_numPoints;
    float4* d_outputImage; // Buffer for the image sent to the screen

    // --- K-Means State ---
    // Configuration
    int m_k;                // Number of clusters
    int m_maxIterations;    // Stop after this many iterations
    int m_currentIteration; // The current iteration number

    // GPU Buffers for the K-Means Algorithm
    float4* d_inputImage;       // Original image pixel colors
    int*    d_clusterIds;   // Assigned cluster for each pixel
    float4* d_centroids;    // Color of each of the K centroids
    float4* d_sums;         // Temporary buffer for summing cluster colors
    int*    d_counts;       // Temporary buffer for counting points in clusters
};
