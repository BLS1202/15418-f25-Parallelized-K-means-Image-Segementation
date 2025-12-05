#pragma once // Good practice to prevent multiple inclusions
class Image;
// HPCRenderer is the abstract base class that defines the interface for all renderers.
class Renderer {
public:
    // The constructor will handle common initialization.
    Renderer(Image* image);
    // A virtual destructor is CRITICAL for a base class.
    // This ensures that when we delete a base class pointer, the
    // correct derived class destructor (e.g., ~CudaRenderer) is called.
    virtual ~Renderer();
    // --- Pure Virtual Functions (The Interface) ---
    // These functions MUST be implemented by any class that inherits from HPCRenderer.
    // Fetches the current rendered image from the device for display.
    virtual Image* getDisplayImage() = 0;
    // Resets the algorithm and picks K random centroids.
    virtual void startKMeansSegmentation() = 0;
    // Executes a single iteration of the K-Means algorithm.
    virtual void stepKMeansIteration() = 0;
    // --- Concrete Function ---
    // This function can be shared by all derived classes without changes.
    virtual bool isKMeansDone() const;
protected: // Use 'protected' so derived classes (CudaRenderer, etc.) can access these.
    
    // --- Common Image Properties ---
    int m_width;
    int m_height;
    int m_numPoints;
    // --- Common K-Means State ---
    int m_k;                // Number of clusters
    int m_maxIterations;    // Stop after this many iterations
    int m_currentIteration; // The current iteration number
};