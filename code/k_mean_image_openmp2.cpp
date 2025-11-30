
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <ctime>
#include <limits>
#include <omp.h>
#include <chrono>
#include <iomanip>

struct Point {
    double r, g, b;
    int clusterId;
};


// 3D Euclidean distance between colors
double distance(Point p1, Point p2) {
    return std::sqrt(std::pow(p1.r - p2.r, 2) + std::pow(p1.g - p2.g, 2) + std::pow(p1.b - p2.b, 2));
}

void save_image_to_ppm(const std::string& filename, const std::vector<unsigned char>& image_data, int width, int height) {
    std::ofstream ppm_file(filename, std::ios::out | std::ios::binary);
    if (!ppm_file) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    ppm_file << "P6\n";
    ppm_file << width << " " << height << "\n";
    ppm_file << "255\n";

    ppm_file.write(reinterpret_cast<const char*>(image_data.data()), image_data.size());
    
    ppm_file.close();
    std::cout << "\nSuccessfully saved image to '" << filename << "'" << std::endl;
}


int main() {
    const auto init_start = std::chrono::steady_clock::now();
    int IMG_WIDTH = 0;
    int IMG_HEIGHT = 0;
    
    const int K = 8; // number of clusters
    const int MAX_ITERATIONS = 20;

    std::cout << "Starting K-Means Color Clustering..." << std::endl;
    std::cout << "  Clusters (K): " << K << std::endl;
    std::cout << "  Max Iterations: " << MAX_ITERATIONS << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    // reading from a file
    std::vector<Point> points;
    std::string inputFilename = "camera_man.ppm"; // PPM file name

    std::ifstream ppm_file(inputFilename, std::ios::in | std::ios::binary);
    if (!ppm_file) {
        std::cerr << "Error: Could not open file '" << inputFilename << "'. Please check the name." << std::endl;
        return 1;
    }
    std::string line;
    int max_val;
    ppm_file >> line; 
    while (ppm_file.peek() == '\n' || ppm_file.peek() == '#') { ppm_file.ignore(256, '\n'); }
    ppm_file >> IMG_WIDTH >> IMG_HEIGHT;
    ppm_file >> max_val;
    ppm_file.ignore(256, '\n');

    std::cout << "Reading image '" << inputFilename << "' (" << IMG_WIDTH << "x" << IMG_HEIGHT << ")" << std::endl;
    
    std::vector<unsigned char> raw_pixel_data(IMG_WIDTH * IMG_HEIGHT * 3);
    ppm_file.read(reinterpret_cast<char*>(raw_pixel_data.data()), raw_pixel_data.size());
    ppm_file.close();

    // Convert raw byte data into point structs
    for (size_t i = 0; i < raw_pixel_data.size(); i += 3) {
        points.push_back({(double)raw_pixel_data[i], (double)raw_pixel_data[i+1], (double)raw_pixel_data[i+2], -1});
    }
    std::cout << "Loaded " << points.size() << " pixels as data points." << std::endl;

    // initialize Centroids
    std::vector<Point> centroids;
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<int> dist(0, points.size() - 1);
    for (int i = 0; i < K; ++i) {
        // pick a random pixel from the image as an initial centroid
        centroids.push_back(points[dist(rng)]);
    }

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();

    // Run K-Means
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        int changed = 0;
        // Assignment Step
        #pragma omp parallel for reduction(||:changed)
        for (int p = 0; p < points.size(); p++) {
            double min_dist = std::numeric_limits<double>::max();
            int closest_centroid_id = -1;
            for (int i = 0; i < K; ++i) {
                double d = distance(points[p], centroids[i]);
                if (d < min_dist) {
                    min_dist = d; 
                    closest_centroid_id = i;
                }
            }
            if(points[p].clusterId != closest_centroid_id){
                points[p].clusterId = closest_centroid_id;
                changed = 1; // some pixels changed its cluster
            }
        }
        
        // Update Step
        std::vector<Point> new_centroids(K, {0, 0, 0, -1}); // {r,g,b,id}
        int r[K] = {0};
        int g[K] = {0};
        int b[K] = {0};
        int counts[K] = {0};

        #pragma omp parallel for reduction(+:r[:K], g[:K], b[:K], counts[:K])
        for (int p = 0; p < points.size(); p++){
            int cluster_id = points[p].clusterId;
            r[cluster_id] += points[p].r;
            g[cluster_id] += points[p].g; 
            b[cluster_id] += points[p].b; 
            counts[cluster_id] += 1;
        }

        #pragma omp parallel for
        for (int i = 0; i < K; i++){
            new_centroids[i].r = r[i];
            new_centroids[i].g = g[i];
            new_centroids[i].b = b[i];
        }
        
        #pragma omp parallel for
        for (int i = 0; i < K; i++) {
            if (counts[i] > 0) {
                centroids[i].r = new_centroids[i].r / counts[i]; 
                centroids[i].g = new_centroids[i].g / counts[i]; 
                centroids[i].b = new_centroids[i].b / counts[i]; 
            }
        }

        // convergence
        if (changed == 0) {
            //std::cout << "Convergence reached at iteration " << iter + 1 << std::endl;
            break;
        } /* else {
            // std::cout << "Iteration " << iter + 1 << " complete." << std::endl;
    
        } */
    }

    // update the image with the new colors
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    const auto updateimage_start = std::chrono::steady_clock::now();
    std::vector<unsigned char> result_image(IMG_WIDTH * IMG_HEIGHT * 3);
    #pragma omp parallel for
    for (int i = 0; i < points.size(); i++) {
        Point centroid_color = centroids[points[i].clusterId];
        result_image[i * 3 + 0] = static_cast<unsigned char>(centroid_color.r);
        result_image[i * 3 + 1] = static_cast<unsigned char>(centroid_color.g);
        result_image[i * 3 + 2] = static_cast<unsigned char>(centroid_color.b);
    }

    const double updateimage_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - updateimage_start).count();
    std::cout << "update image time (sec): " << updateimage_time << '\n';

    std::cout << "Image data stored in vector." << std::endl;


    // write the array to a file
    save_image_to_ppm("kmeans_quantized.ppm", result_image, IMG_WIDTH, IMG_HEIGHT);


    return 0;
}