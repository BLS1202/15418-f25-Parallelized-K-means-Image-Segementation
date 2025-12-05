#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "../Image.h"
#include "../Renderer.h"
#include <cuda_runtime.h>

class CudaRenderer : public Renderer{
public:
    // Constructor / Destructor
    CudaRenderer(Image* image, int k = 8);
    virtual ~CudaRenderer();

    // HPCRenderer interface
    void startKMeansSegmentation() override;
    void stepKMeansIteration() override;
    bool isKMeansDone() const override;
    Image* getDisplayImage() override;

private:
    // Image dimensions and K
    int m_width;
    int m_height;
    int m_numPoints;
    int m_k;

    // Iteration control
    int m_currentIteration;
    int m_maxIterations;

    // Host-side display image
    Image* m_displayImage;

    // GPU buffers
    float4* d_inputImage;
    float4* d_outputImage;
    float4* d_centroids;
    float4* d_centroid_sums;
    int*    d_counts;
    int*    d_clusterIds;

    // CUDA block/grid configuration
    dim3 m_blockDim;
    dim3 m_gridDim;
};

#endif // CUDA_RENDERER_H

