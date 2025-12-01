#include <iostream>
#include "platformgl.h" // For OpenGL/GLUT headers
#include "Image.h"
#include "HPCRenderer.h"  // Use the HPC Renderer

// --- Global Variables ---
HPCRenderer* g_renderer = nullptr;
Image* g_displayImage = nullptr;

const int DELAY_BETWEEN_ITERATIONS_MS = 500;

// --- Forward Declarations ---
void runNextClusteringStep(int value);
void cleanup();

// --- GLUT Callbacks ---

void handleDisplay() {
    if (!g_displayImage || !g_displayImage->data) return;

    // The display image is on the host, so draw it directly.
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

// The timer function that drives the k-means animation.
void runNextClusteringStep(int value) {
    if (g_renderer && !g_renderer->isKMeansDone()) {
        // Execute one iteration of the K-Means algorithm on the GPU.
        g_renderer->stepKMeansIteration();
        
        // Free the old host image memory.
        delete g_displayImage;

        // Fetch the newly updated image from the GPU for display.
        g_displayImage = g_renderer->getDisplayImage();

        // Request a redraw to show the changes.
        glutPostRedisplay();

        // Schedule this function to run again after a delay.
        glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);

    } else {
        std::cout << "\nK-Means clustering complete. Press 'q' to quit." << std::endl;
    }
}


// --- Main Application Entry Point ---
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image.ppm>" << std::endl;
        return 1;
    }
    
    // Load the original image from disk.
    Image* originalImage = new Image(0, 0);
    if (!originalImage->loadPPM(argv[1])) {
        delete originalImage;
        return 1;
    }
    
    // Create the HPCRenderer, which copies the image data to the GPU.
    g_renderer = new HPCRenderer(originalImage);
    int width = originalImage->width;
    int height = originalImage->height;
    // Standard GLUT setup.
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("HPC K-Means Clustering Visualization");

    // Once data is on the GPU, the initial host copy is no longer needed.
    delete originalImage;

    glutDisplayFunc(handleDisplay);
    glutKeyboardFunc(handleKeyPress);

    // Initialize the K-Means algorithm on the GPU.
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

