#include "Renderer.h"
#include "Image.h"

Renderer::Renderer(Image* image) {
    m_width = image->width;
    m_height = image->height;
    m_numPoints = m_width * m_height;
}

Renderer::~Renderer() {
    // Nothing to do in the base class destructor, but it must be defined.
}

