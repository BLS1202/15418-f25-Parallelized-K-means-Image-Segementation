// #include <iostream>
// #include "platformgl.h" // For OpenGL/GLUT headers
// #include "Image.h"
// #include "SimpleRenderer.h"

// // Global renderer pointer
// SimpleRenderer* renderer = nullptr;
// int window_width = 0;
// int window_height = 0;

// void handleDisplay() {
//     // Tell the renderer to execute its CUDA kernel
//     renderer->render();

//     // Get the resulting image (after it's been copied back to the CPU)
//     const Image* img = renderer->getDisplayImage();

//     // Use OpenGL to draw the raw pixel data to the screen
//     glClear(GL_COLOR_BUFFER_BIT);
//     glDrawPixels(img->width, img->height, GL_RGBA, GL_FLOAT, img->data);
//     glutSwapBuffers();
// }

// void handleKeyPress(unsigned char key, int x, int y) {
//     if (key == 'q' || key == 'Q') {
//         delete renderer;
//         exit(0);
//     }
// }

// int main(int argc, char** argv) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <input_image.ppm>" << std::endl;
//         return 1;
//     }
    
//     // Load the image from disk
//     Image hostImage(0, 0);
//     if (!hostImage.loadPPM(argv[1])) {
//         return 1;
//     }
//     window_width = hostImage.width;
//     window_height = hostImage.height;

//     Image* segmentedImage = nullptr;
//     {
//         // We create a temporary renderer on the stack.
//         // It will be automatically destroyed when this block ({}) ends.
//         SimpleRenderer processor(&hostImage);
//         segmentedImage = processor.segmentationImage(&hostImage);
//     } // processor is destroyed here, but `segmentedImage` pointer remains.
//     if (!segmentedImage) {
//         std::cerr << "Error: Segmentation failed to produce an image." << std::endl;
//         return 1;
//     }
    
//     renderer = new SimpleRenderer(segmentedImage);
    
//     // =================================================================
//     // STEP 4: Standard GLUT setup, using the dimensions of our
//     //         NEW processed image.
//     // =================================================================
//     glutInit(&argc, argv);
//     glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
//     glutInitWindowSize(segmentedImage->width, segmentedImage->height);
//     glutCreateWindow("K-Means Segmentation Result");
//     glutDisplayFunc(handleDisplay);
//     glutKeyboardFunc(handleKeyPress);
//     std::cout << "Window created. Displaying segmented image. Press 'q' to quit." << std::endl;
    
//     glutMainLoop();
//     // Although glutMainLoop never returns, it's good practice
//     // delete renderer; // This would be here if the loop could exit.
//     return 0;

//     // // Initialize our renderer with the loaded image data
//     // renderer = new SimpleRenderer(&hostImage);
    
//     // // Standard GLUT setup
//     // glutInit(&argc, argv);
//     // glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
//     // glutInitWindowSize(window_width, window_height);
//     // glutCreateWindow("Simple CUDA Image Renderer");

//     // glutDisplayFunc(handleDisplay);
//     // glutKeyboardFunc(handleKeyPress);

//     // std::cout << "Window created. Press 'q' to quit." << std::endl;
    
//     // glutMainLoop();

//     // return 0;
// }

#include <iostream>
#include "platformgl.h" // For OpenGL/GLUT headers
#include "Image.h"
#include "SimpleRenderer.h"

// --- Global variables for automation and display ---
SimpleRenderer* g_renderer = nullptr;   // The renderer responsible for displaying the *current* image
Image* g_originalImage = nullptr;       // A pointer to the original, unmodified image

const int TOTAL_ITERATIONS = 5;
const int DELAY_BETWEEN_ITERATIONS_MS = 100;
int currentIteration = 0;

// --- Forward declaration for our timer function ---
void runNextClusteringStep(int value);

// --- Display function ---
// THIS IS THE EXACT SAME LOGIC FROM YOUR WORKING VERSION.
// It uses the global renderer to handle the display pipeline.
void handleDisplay() {
    if (!g_renderer) return;

    // Tell the renderer to execute its CUDA kernel (GPU copy)
    g_renderer->render();

    // Get the resulting image (after it's been copied back to the CPU)
    const Image* img = g_renderer->getDisplayImage();

    // Use OpenGL to draw the raw pixel data to the screen
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(img->width, img->height, GL_RGBA, GL_FLOAT, img->data);
    glutSwapBuffers();
}

// --- Keyboard handler for quitting ---
// Now cleans up both global pointers.
void handleKeyPress(unsigned char key, int x, int y) {
    if (key == 'q' || key == 'Q') {
        delete g_renderer;
        delete g_originalImage;
        exit(0);
    }
}

// --- The Timer Function for Automation ---
void runNextClusteringStep(int value) {
    if (currentIteration < TOTAL_ITERATIONS) {
        currentIteration++;
        std::cout << "--- Starting automatic clustering run " << currentIteration << " of " << TOTAL_ITERATIONS << " ---" << std::endl;

        // 1. Create a temporary processor to run the CPU-based segmentation
        //    We always run it on the original image.
        SimpleRenderer tempProcessor(g_originalImage);
        Image* newSegmentedImage = tempProcessor.segmentationImage(g_originalImage);

        if (newSegmentedImage) {
            // 2. Delete the OLD display renderer to free its GPU memory.
            delete g_renderer;

            // 3. Create a NEW display renderer, initialized with our new processed image.
            //    This correctly sends the new image data to the GPU.
            g_renderer = new SimpleRenderer(newSegmentedImage);

            // 4. The image data is now managed by the renderer, so we can delete
            //    the CPU copy that segmentationImage() returned.
            delete newSegmentedImage;

            // 5. Tell GLUT to redraw the window with the new image.
            glutPostRedisplay();
        }

        // 6. Schedule this function to run again.
        glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);

    } else {
        std::cout << "\n>>> Automation complete. Final image is displayed. <<<" << std::endl;
        std::cout << "Press 'q' to quit." << std::endl;
    }
}


// --- Main function ---
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image.ppm>" << std::endl;
        return 1;
    }
    
    // Load the original image from disk into our global pointer
    g_originalImage = new Image(0, 0);
    if (!g_originalImage->loadPPM(argv[1])) {
        delete g_originalImage;
        return 1;
    }
    
    // Create the initial renderer to display the ORIGINAL image first.
    g_renderer = new SimpleRenderer(g_originalImage);
    
    // Standard GLUT setup
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(g_originalImage->width, g_originalImage->height);
    glutCreateWindow("Automated K-Means Clustering");

    glutDisplayFunc(handleDisplay);
    glutKeyboardFunc(handleKeyPress);
    
    std::cout << "Displaying original image. Automated processing will begin shortly..." << std::endl;
    
    // Kick off the timer to start the first processing step after a delay.
    glutTimerFunc(DELAY_BETWEEN_ITERATIONS_MS, runNextClusteringStep, 0);
    
    glutMainLoop();

    return 0;
}
