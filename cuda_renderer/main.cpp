// #include <iostream>
// #include "platformgl.h" // For OpenGL/GLUT headers
// #include "Image.h"
// #include "HPCRenderer.h"  // Use the HPC Renderer

// // --- Global Variables ---
// HPCRenderer* g_renderer = nullptr;
// Image* g_displayImage = nullptr;

// const int DELAY_BETWEEN_ITERATIONS_MS = 500;

// // --- Forward Declarations ---
// void runNextClusteringStep(int value);
// void cleanup();

// // --- GLUT Callbacks ---

// void handleDisplay() {
//     if (!g_displayImage || !g_displayImage->data) return;

//     // The display image is on the host, so draw it directly.
//     glClear(GL_COLOR_BUFFER_BIT);
//     glDrawPixels(g_displayImage->width, g_displayImage->height, GL_RGBA, GL_FLOAT, g_displayImage->data);
//     glutSwapBuffers();
// }

// void handleKeyPress(unsigned char key, int x, int y) {
//     if (key == 'q' || key == 'Q') {
//         cleanup();
//         exit(0);
//     }
// }

// void cleanup() {
//     delete g_renderer;
//     delete g_displayImage;
//     g_renderer = nullptr;
//     g_displayImage = nullptr;
// }

// // The timer function that drives the k-means animation.
// void runNextClusteringStep(int value) {
//     if (g_renderer && !g_renderer->isKMeansDone()) {
//         // Execute one iteration of the K-Means algorithm on the GPU.
//         g_renderer->stepKMeansIteration();
        
//         // Free the old host image memory.
//         delete g_displayImage;

//         // Fetch the newly updated image from the GPU for display.
//         g_displayImage = g_renderer->getDisplayImage();

//         // Request a redraw to show the changes.
//         glutPostRedisplay();

//         // Schedule this function to run again after a delay.
//         glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);

//     } else {
//         std::cout << "\nK-Means clustering complete. Press 'q' to quit." << std::endl;
//     }
// }


// // --- Main Application Entry Point ---
// int main(int argc, char** argv) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <input_image.ppm>" << std::endl;
//         return 1;
//     }
    
//     // Load the original image from disk.
//     Image* originalImage = new Image(0, 0);
//     if (!originalImage->loadPPM(argv[1])) {
//         delete originalImage;
//         return 1;
//     }
    
//     // Create the HPCRenderer, which copies the image data to the GPU.
//     g_renderer = new HPCRenderer(originalImage);
//     int width = originalImage->width;
//     int height = originalImage->height;
//     // Standard GLUT setup.
//     glutInit(&argc, argv);
//     glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
//     glutInitWindowSize(width, height);
//     glutCreateWindow("HPC K-Means Clustering Visualization");

//     // Once data is on the GPU, the initial host copy is no longer needed.
//     delete originalImage;

//     glutDisplayFunc(handleDisplay);
//     glutKeyboardFunc(handleKeyPress);

//     // Initialize the K-Means algorithm on the GPU.
//     g_renderer->startKMeansSegmentation();
    
//     // Create an empty image buffer for the first frame.
//     g_displayImage = g_renderer->getDisplayImage();
    
//     std::cout << "Starting K-Means clustering process..." << std::endl;
    
//     // Kick off the timer to start the iterative process.
//     glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);
    
//     // Enter the GLUT event processing loop.
//     glutMainLoop(); 

//     cleanup();
//     return 0;
// }

#include <iostream>
#include <string> // Needed for comparing renderer type
#include "platformgl.h" 
#include "Image.h"
// --- Include all available renderer headers ---
#include "SimpleRenderer/HPCRenderer.h"
#include "renderer_cudd/CudaRenderer.h"
#include "renderer_openMP/OpenMPRenderer.h"
// --- Global Variables ---
// g_renderer must be a pointer to the BASE CLASS so it can hold any of the derived renderers.
HPCRenderer* g_renderer = nullptr;
Image* g_displayImage = nullptr;
const int DELAY_BETWEEN_ITERATIONS_MS = 500;
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
// --- Main Application Entry Point (MODIFIED) ---
int main(int argc, char** argv) {
    // We now need 3 arguments: the program name, the renderer type, and the image path.
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <renderer> <input_image.ppm>" << std::endl;
        std::cerr << "  Available renderers: simple, openmp, cuda" << std::endl;
        return 1;
    }
    // --- 1. Argument Parsing ---
    std::string rendererType = argv[1];
    std::string imagePath = argv[2];
    // Load the original image from disk.
    Image* originalImage = new Image(0, 0);
    if (!originalImage->loadPPM(imagePath)) {
        delete originalImage;
        return 1;
    }
    // --- 2. Conditional Renderer Creation ---
    // The global g_renderer (HPCRenderer*) will point to the specific object we create.
    if (rendererType == "simple") {
        std::cout << "Using Simple (CPU) Renderer..." << std::endl;
        g_renderer = new HPCRenderer(originalImage);
    } else if (rendererType == "openmp") {
        std::cout << "Using OpenMP Renderer..." << std::endl;
        g_renderer = new OpenMPRenderer(originalImage);
    } else if (rendererType == "cuda") {
        std::cout << "Using CUDA Renderer..." << std::endl;
        g_renderer = new CudaRenderer(originalImage);
    } else {
        std::cerr << "Error: Unknown renderer type '" << rendererType << "'." << std::endl;
        std::cerr << "Please choose from: simple, openmp, cuda" << std::endl;
        delete originalImage;
        return 1;
    }
    // --- 3. Standard GLUT and Application Setup (Unchanged) ---
    int width = originalImage->width;
    int height = originalImage->height;
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("HPC K-Means Clustering Visualization");
    // The host copy of the image is no longer needed after the renderer is created.
    delete originalImage;
    glutDisplayFunc(handleDisplay);
    glutKeyboardFunc(handleKeyPress);
    // Initialize K-Means on whichever device was chosen.
    g_renderer->startKMeansSegmentation();
    
    // Create an empty image buffer for the first frame.
    g_displayImage = g_renderer->getDisplayImage();
    
    std::cout << "Starting K-Means clustering process..." << std::endl;
    
    // Kick off the timer to start the iterative process.
    glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);
    
    // Enter the GLUT event processing loop.
    glutMainLoop(); 
    cleanup();
    return 0;
}