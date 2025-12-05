#pragma once
#include "../Renderer.h"
#include <vector> 

struct Point;
class Image;
struct float4;

class HPCRenderer : public Renderer {
public:

    HPCRenderer(Image* image);


    virtual ~HPCRenderer();

    Image* getDisplayImage() override;




    void startKMeansSegmentation() override;

    void stepKMeansIteration() override;

     bool isKMeansDone() const override;

private:
    // --- Display & Image Properties ---
    int m_width;
    int m_height;
    int m_numPoints;
    float4* d_inputImage;  // Original image data on the GPU
    float4* d_outputImage; // Buffer for the image sent to the screen

    // --- K-Means State ---
    int m_k;                // Number of clusters
    int m_maxIterations;    // Stop after this many iterations
    int m_currentIteration; // The current iteration number
    bool m_converged;       // Flag to check for convergence


    std::vector<Point> m_points;
    std::vector<Point> m_centroids;
};
