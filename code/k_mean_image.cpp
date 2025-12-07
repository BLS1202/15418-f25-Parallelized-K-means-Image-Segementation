#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <ctime>
#include <limits>
#include <chrono>
#include <iomanip>
#include "libraries/Image.h"

struct Point {
    double r, g, b; // Using double for precision in centroid calculations
    int clusterId;
};


double distance(Point p1, Point p2) {
    return std::sqrt(std::pow(p1.r - p2.r, 2) + std::pow(p1.g - p2.g, 2) + std::pow(p1.b - p2.b, 2));
}



int main(int argc, char** argv) {
    const auto init_start = std::chrono::steady_clock::now();
    int IMG_WIDTH = 0;
    int IMG_HEIGHT = 0;
    

    const int K = 8; // Number of clusters (dominant colors)
    const int MAX_ITERATIONS = 20;

    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << "<input_image.jpg>" << "<output_image.jpg>" <<std::endl;
        return 1;
    }

    std::string imagePath = argv[1];
    std::string outPath = argv[2];
    Image* originalImage = new Image(0, 0);
    if (!originalImage->loadJPG(imagePath)) {
        delete originalImage;
        return 1;
    }

    std::cout << "Starting K-Means Color Clustering..." << std::endl;
    std::cout << "  Clusters (K): " << K << std::endl;
    std::cout << "  Max Iterations: " << MAX_ITERATIONS << std::endl;
    std::cout << "------------------------------------" << std::endl;

    
    std::vector<Point> points;
    IMG_WIDTH = originalImage->width;   
    IMG_HEIGHT = originalImage->height; 

    float* img_data = originalImage -> data;
    size_t total_pixels = (size_t)IMG_WIDTH * IMG_HEIGHT;

    for (size_t i = 0; i < total_pixels; ++i) {

        size_t data_index = i * 4; 

        double r = (double)(img_data[data_index + 0] * 255.0f); 
        double g = (double)(img_data[data_index + 1] * 255.0f); 
        double b = (double)(img_data[data_index + 2] * 255.0f); 

        points.push_back({r, g, b, -1});
    }

    std::cout << "Loaded " << points.size() << " pixels as data points." << std::endl;

    // 2. Initialize Centroids
    std::vector<Point> centroids;
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, points.size() - 1);
    for (int i = 0; i < K; ++i) {
        // Pick a random pixel from the image as an initial centroid
        centroids.push_back(points[dist(rng)]);
    }

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();

    // 3. Run K-Means (The core loop is UNCHANGED logically!)
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        bool changed = false;
        // Assignment Step
        for (auto& point : points) {
            double min_dist = std::numeric_limits<double>::max();
            int closest_centroid_id = -1;
            for (int i = 0; i < K; ++i) {
                double d = distance(point, centroids[i]);
                if (d < min_dist) { min_dist = d; closest_centroid_id = i; }
            }
            if (point.clusterId != closest_centroid_id) {
                point.clusterId = closest_centroid_id;
                changed = true;
            }
        }
        // Update Step
        std::vector<Point> new_centroids(K, {0, 0, 0, -1}); // >> Changed to {r,g,b,id}
        std::vector<int> counts(K, 0);
        for (const auto& point : points) {
            int cluster_id = point.clusterId;
            new_centroids[cluster_id].r += point.r; // >> Changed from .x to .r
            new_centroids[cluster_id].g += point.g; // >> Changed from .y to .g
            new_centroids[cluster_id].b += point.b; // >> ADDED for blue channel <<
            counts[cluster_id]++;
        }
        for (int i = 0; i < K; ++i) {
            if (counts[i] > 0) {
                centroids[i].r = new_centroids[i].r / counts[i]; // >> Changed from .x to .r
                centroids[i].g = new_centroids[i].g / counts[i]; // >> Changed from .y to .g
                centroids[i].b = new_centroids[i].b / counts[i]; // >> ADDED for blue channel <<
            }
        }
        // Convergence Check
        if (!changed) {
            //std::cout << "Convergence reached at iteration " << iter + 1 << std::endl;
            break;
        } /* else {
            std::cout << "Iteration " << iter + 1 << " complete." << std::endl;
        } */
    }
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    const auto updateimage_start = std::chrono::steady_clock::now();
    std::vector<unsigned char> result_image(IMG_WIDTH * IMG_HEIGHT * 3);
    for (size_t i = 0; i < points.size(); ++i) {
        // Get the centroid color for the current pixel
        Point centroid_color = centroids[points[i].clusterId];
        
        // Write the RGB values to the image data array
        result_image[i * 3 + 0] = static_cast<unsigned char>(centroid_color.r);
        result_image[i * 3 + 1] = static_cast<unsigned char>(centroid_color.g);
        result_image[i * 3 + 2] = static_cast<unsigned char>(centroid_color.b);

    }

    const double updateimage_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - updateimage_start).count();
    std::cout << "update image time (sec): " << updateimage_time << '\n';

    std::cout << "Image data stored in vector." << std::endl;


    // Call the save function to write the array to a file
    originalImage -> save_image_to_jpg(outPath, result_image, IMG_WIDTH, IMG_HEIGHT, 90);


    return 0;
}
