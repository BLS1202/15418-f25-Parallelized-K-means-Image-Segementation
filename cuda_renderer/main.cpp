#include <iostream>
#include "platformgl.h" // For OpenGL/GLUT headers
#include "Image.h"
#include "SimpleRenderer.h"

// Global renderer pointer
SimpleRenderer* renderer = nullptr;
int window_width = 0;
int window_height = 0;

void handleDisplay() {
    // Tell the renderer to execute its CUDA kernel
    renderer->render();

    // Get the resulting image (after it's been copied back to the CPU)
    const Image* img = renderer->getDisplayImage();

    // Use OpenGL to draw the raw pixel data to the screen
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(img->width, img->height, GL_RGBA, GL_FLOAT, img->data);
    glutSwapBuffers();
}

void handleKeyPress(unsigned char key, int x, int y) {
    if (key == 'q' || key == 'Q') {
        delete renderer;
        exit(0);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image.ppm>" << std::endl;
        return 1;
    }
    
    // Load the image from disk
    Image hostImage(0, 0);
    if (!hostImage.loadPPM(argv[1])) {
        return 1;
    }
    window_width = hostImage.width;
    window_height = hostImage.height;

    Image* segmentedImage = nullptr;
    {
        // We create a temporary renderer on the stack.
        // It will be automatically destroyed when this block ({}) ends.
        SimpleRenderer processor(&hostImage);
        segmentedImage = processor.segmentationImage(&hostImage);
    } // processor is destroyed here, but `segmentedImage` pointer remains.
    if (!segmentedImage) {
        std::cerr << "Error: Segmentation failed to produce an image." << std::endl;
        return 1;
    }
    
    renderer = new SimpleRenderer(segmentedImage);
    
    // =================================================================
    // STEP 4: Standard GLUT setup, using the dimensions of our
    //         NEW processed image.
    // =================================================================
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(segmentedImage->width, segmentedImage->height);
    glutCreateWindow("K-Means Segmentation Result");
    glutDisplayFunc(handleDisplay);
    glutKeyboardFunc(handleKeyPress);
    std::cout << "Window created. Displaying segmented image. Press 'q' to quit." << std::endl;
    
    glutMainLoop();
    // Although glutMainLoop never returns, it's good practice
    // delete renderer; // This would be here if the loop could exit.
    return 0;

    // // Initialize our renderer with the loaded image data
    // renderer = new SimpleRenderer(&hostImage);
    
    // // Standard GLUT setup
    // glutInit(&argc, argv);
    // glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    // glutInitWindowSize(window_width, window_height);
    // glutCreateWindow("Simple CUDA Image Renderer");

    // glutDisplayFunc(handleDisplay);
    // glutKeyboardFunc(handleKeyPress);

    // std::cout << "Window created. Press 'q' to quit." << std::endl;
    
    // glutMainLoop();

    // return 0;
}