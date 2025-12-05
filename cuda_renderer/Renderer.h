#pragma once
class Image;
// HPCRenderer is the abstract base class that defines the interface for all renderers.
class Renderer {
public:

    Renderer(Image* image);

    virtual ~Renderer();

    virtual Image* getDisplayImage() = 0;

    virtual void startKMeansSegmentation() = 0;

    virtual void stepKMeansIteration() = 0;

    virtual bool isKMeansDone() const = 0;
protected: 
    
    // --- Common Image Properties ---
    int m_width;
    int m_height;
    int m_numPoints;
    // --- Common K-Means State ---
    int m_k;                // Number of clusters
    int m_maxIterations;    // Stop after this many iterations
    int m_currentIteration; // The current iteration number
};