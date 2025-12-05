#include <iostream>
#include <string>
#include "platformgl.h" 
#include "Image.h"

#include "SimpleRenderer/HPCRenderer.h"
#include "renderer_openMP/OpenMPrenderer.h"
#include "renderer_cuda/CUDARenderer.h"

// --- Global Variables ---
Renderer* g_renderer = nullptr; 
Image* g_displayImage = nullptr;

const int DELAY_BETWEEN_ITERATIONS_MS = 250;

// --- Forward Declarations ---
void runNextClusteringStep(int value);
void cleanup();

// --- GLUT Callbacks (Unchanged) ---
void handleDisplay() {
    if (!g_displayImage || !g_displayImage->data) return;
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(g_displayImage->width, g_displayImage->height, GL_RGBA, GL_FLOAT, g_displayImage->data);
    glutSwapBuffers();
}

void handleKeyPress(unsigned char key, int x, int y) {
    if (key == 'q' || key == 'Q') {
        cleanup();
        exit(0);
    }
}

void cleanup() {
    delete g_renderer;
    delete g_displayImage;
    g_renderer = nullptr;
    g_displayImage = nullptr;
}

void runNextClusteringStep(int value) {
    if (g_renderer && !g_renderer->isKMeansDone()) {
        g_renderer->stepKMeansIteration();
        delete g_displayImage;
        g_displayImage = g_renderer->getDisplayImage();
        glutPostRedisplay();
        glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);
    } else {
        std::cout << "\nK-Means clustering complete. Press 'q' to quit." << std::endl;
    }
}

// --- Main Application Entry Point ---
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <renderer> <input_image.ppm>" << std::endl;
        std::cerr << "  Available renderers: simple, openmp, cuda" << std::endl;
        return 1;
    }
    
    std::string rendererType = argv[1];
    std::string imagePath = argv[2];

    Image* originalImage = new Image(0, 0);
    if (!originalImage->loadPPM(imagePath)) {
        delete originalImage;
        return 1;
    }

    // --- Conditional Renderer Creation ---
    if (rendererType == "simple") {
        std::cout << "Using Simple (CPU) Renderer..." << std::endl;
        g_renderer = new HPCRenderer(originalImage);
    } else if (rendererType == "openmp") {
        std::cout << "Using OpenMP Renderer..." << std::endl;
        g_renderer = new OpenMPRenderer(originalImage);
    } else if (rendererType == "cuda") {
        // You would need to create CUDARenderer.h and .cu for this to link
        std::cout << "Using CUDA Renderer..." << std::endl;
        g_renderer = new CUDARenderer(originalImage); 
    } else {
        std::cerr << "Error: Unknown renderer type '" << rendererType << "'." << std::endl;
        std::cerr << "Please choose from: simple, openmp, cuda" << std::endl;
        delete originalImage;
        return 1;
    }

    int width = originalImage->width;
    int height = originalImage->height;
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("HPC K-Means Clustering Visualization");
    
    delete originalImage;

    glutDisplayFunc(handleDisplay);
    glutKeyboardFunc(handleKeyPress);
    
    g_renderer->startKMeansSegmentation();
    g_displayImage = g_renderer->getDisplayImage();
    
    std::cout << "Starting K-Means clustering process..." << std::endl;
    
    glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);
    
    glutMainLoop(); 
    cleanup();
    return 0;
}
