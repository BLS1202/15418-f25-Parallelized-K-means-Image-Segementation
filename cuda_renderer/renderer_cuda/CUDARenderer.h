#pragma once

#include "../Renderer.h"
#include <vector>


class Image;

class CUDARenderer : public Renderer {s
public:

    CUDARenderer(Image* image);


    ~CUDARenderer();


    Image* getDisplayImage() override;



    void startKMeansSegmentation() override;


    void stepKMeansIteration() override;


    bool isKMeansDone() const override;

private:
    int m_width;
    int m_height;
    int m_numPoints;


    int m_k;                // Number of clusters
    int m_maxIterations;    // Stop after this many iterations
    int m_currentIteration; // The current iteration number


    float4* d_inputImage;
    float4* d_outputImage;
    float4* d_centroids;
    float4* d_centroid_sums;
    int*    d_counts;
    

    std::vector<float4> h_input_points;
};

