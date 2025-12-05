
#include <iostream>
#include "../platformgl.h"
#include "../Image.h"
#include "CUDARenderer.h"

// --- Global Variables ---
Renderer* g_renderer = nullptr;
Image* g_displayImage = nullptr;
const int DELAY_BETWEEN_ITERATIONS_MS = 500;

// --- Forward Declarations ---
void runNextClusteringStep(int value);
void cleanup();

// --- GLUT Callbacks ---
void handleDisplay() {
    if (!g_displayImage || !g_displayImage->data) return;

    glClear(GL_COLOR_BUFFER_BIT);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // Ensure correct row alignment

    glDrawPixels(g_displayImage->width, g_displayImage->height,
                 GL_RGBA, GL_FLOAT, g_displayImage->data);

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
        // delete g_displayImage;   <-- REMOVE THIS
        g_displayImage = g_renderer->getDisplayImage(); // safe, points to internal image
        glutPostRedisplay();
        glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);
    } else {
        std::cout << "\nK-Means clustering complete. Press 'q' to quit." << std::endl;
    }
}


// --- Main ---
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <renderer> <input_image.ppm>" << std::endl;
        return 1;
    }

    std::string rendererType = argv[1];
    std::string imagePath = argv[2];

    // Load image
    std::cout << "here" << std::endl;
    Image* originalImage = new Image(0, 0);
    if (!originalImage->loadPPM(imagePath)) {
        delete originalImage;
        return 1;
    }

    if (rendererType == "cuda") {
        std::cout << "Using CUDA Renderer..." << std::endl;
        g_renderer = new CudaRenderer(originalImage); // keep image alive until after GPU copy
    } else {
        std::cerr << "Error: Unknown renderer type '" << rendererType << "'." << std::endl;
        delete originalImage;
        return 1;
    }

    int width = originalImage->width;
    int height = originalImage->height;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA K-Means Clustering Visualization");

    // Now we can safely delete original image
    delete originalImage;

    glutDisplayFunc(handleDisplay);
    glutKeyboardFunc(handleKeyPress);

    g_renderer->startKMeansSegmentation();
    g_displayImage = g_renderer->getDisplayImage();

    std::cout << "Starting K-Means clustering process..." << std::endl;
    glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);
    std::cout << "check" << std::endl;
    glutMainLoop();

    cleanup();
    return 0;
}
