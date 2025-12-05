#pragma once
#include "../Renderer.h"
#include <vector> 

struct Point;
class Image;
struct float4;
class OpenMPRenderer : public Renderer {
public:

    OpenMPRenderer(Image* image);

    virtual ~OpenMPRenderer();

    Image* getDisplayImage() override;

    void startKMeansSegmentation() override;

    void stepKMeansIteration() override;

    bool isKMeansDone() const override;
private:
    int m_width;
    int m_height;
    int m_numPoints;
    float4* d_inputImage;  // Original image data on the GPU
    float4* d_outputImage; // Buffer for the image sent to the screen

    int m_k;                // Number of clusters
    int m_maxIterations;    // Stop after this many iterations
    int m_currentIteration; // The current iteration number
    bool m_converged;       // Flag to check for convergence

    std::vector<Point> m_points;
    std::vector<Point> m_centroids;
};