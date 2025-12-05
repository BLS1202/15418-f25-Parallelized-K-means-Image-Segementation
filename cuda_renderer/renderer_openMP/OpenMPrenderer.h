#pragma once
#include "../Renderer.h"

class Image;

struct float4;
class OpenMPRenderer : public Renderer {
public:
    // Constructor: Allocates all GPU memory for display and K-Means.
    OpenMPRenderer(Image* image);

    // Destructor: Frees all allocated GPU memory.
    virtual ~OpenMPRenderer();

    // Fetches the current rendered image from the GPU for display.
    // The caller is responsible for deleting the returned Image object.
    Image* getDisplayImage();

    // --- K-Means Control Functions ---

    // Resets the algorithm, picks K random centroids, and resets the iteration count.
    void startKMeansSegmentation();

    // Executes a single iteration of the K-Means algorithm.
    void stepKMeansIteration();

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
};
