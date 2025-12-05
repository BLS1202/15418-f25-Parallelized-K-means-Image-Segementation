#include "Renderer.h"
#include "Image.h"

// Constructor
Renderer::Renderer(Image* image)
{
    if (image) {
        m_width = image->width;
        m_height = image->height;
        m_numPoints = m_width * m_height;
    } else {
        m_width = 0;
        m_height = 0;
        m_numPoints = 0;
    }

    m_k = 0;
    m_maxIterations = 0;
    m_currentIteration = 0;
}

// Virtual destructor
Renderer::~Renderer() {}

// Default isKMeansDone
bool Renderer::isKMeansDone() const
{
    return m_currentIteration >= m_maxIterations;
}
