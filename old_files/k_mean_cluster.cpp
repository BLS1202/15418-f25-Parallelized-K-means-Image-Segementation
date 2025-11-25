
// Filename: kmeans_no_opencv.cpp

#include <iostream>
#include <vector>
#include <cmath>    // For sqrt and pow
#include <random>   // For modern C++ random number generation
#include <ctime>    // To seed the random number generator
#include <limits>   // For std::numeric_limits

// A simple structure to represent a 2D point
struct Point {
    double x, y;
    int clusterId; // The ID of the cluster this point belongs to
};

// Function to calculate the Euclidean distance between two points
double distance(Point p1, Point p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

int main() {
    // --- Configuration ---
    const int NUM_POINTS = 200;
    const int K = 4; // Number of clusters
    const int MAX_ITERATIONS = 20;

    std::cout << "Starting K-Means Clustering..." << std::endl;
    std::cout << "Number of points: " << NUM_POINTS << std::endl;
    std::cout << "Number of clusters (K): " << K << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // --- 1. Generate Random Data Points ---
    std::vector<Point> points;
    // Use modern C++ for better random number generation
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_real_distribution<double> dist(0.0, 100.0); // Points in a 100x100 space

    for (int i = 0; i < NUM_POINTS; ++i) {
        points.push_back({dist(rng), dist(rng), -1}); // {x, y, initial clusterId}
    }

    // --- 2. Initialize Centroids ---
    // A simple way is to pick K random points from the dataset as initial centroids.
    std::vector<Point> centroids;
    std::uniform_int_distribution<int> point_dist(0, NUM_POINTS - 1);
    for (int i = 0; i < K; ++i) {
        int randomIndex = point_dist(rng);
        centroids.push_back(points[randomIndex]);
        centroids[i].clusterId = i;
    }

    // --- 3. Run K-Means Algorithm ---
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        bool changed = false;

        // --- Assignment Step ---
        // For each point, find the closest centroid
        for (auto& point : points) {
            double min_dist = std::numeric_limits<double>::max();
            int closest_centroid_id = -1;

            for (int i = 0; i < K; ++i) {
                double d = distance(point, centroids[i]);
                if (d < min_dist) {
                    min_dist = d;
                    closest_centroid_id = i;
                }
            }
            
            // If the point's cluster assignment changes, note it
            if (point.clusterId != closest_centroid_id) {
                point.clusterId = closest_centroid_id;
                changed = true;
            }
        }

        // --- Update Step ---
        // Recalculate centroids based on the mean of assigned points
        std::vector<Point> new_centroids(K, {0, 0, -1});
        std::vector<int> counts(K, 0);

        for (const auto& point : points) {
            int cluster_id = point.clusterId;
            new_centroids[cluster_id].x += point.x;
            new_centroids[cluster_id].y += point.y;
            counts[cluster_id]++;
        }

        for (int i = 0; i < K; ++i) {
            if (counts[i] > 0) {
                centroids[i].x = new_centroids[i].x / counts[i];
                centroids[i].y = new_centroids[i].y / counts[i];
            }
        }

        // --- Convergence Check ---
        // If no points changed cluster in this iteration, we have converged.
        if (!changed) {
            std::cout << "Convergence reached at iteration " << iter + 1 << std::endl;
            break;
        }
        
        if (iter == MAX_ITERATIONS - 1) {
            std::cout << "Maximum iterations reached." << std::endl;
        }
    }

    // --- 4. Output Results ---
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Final Cluster Assignments:" << std::endl;
    for (int i = 0; i < K; ++i) {
        std::cout << "\nCluster " << i << " (Centroid at " << centroids[i].x << ", " << centroids[i].y << "):" << std::endl;
        for (const auto& point : points) {
            if (point.clusterId == i) {
                std::cout << "  Point(" << point.x << ", " << point.y << ")" << std::endl;
            }
        }
    }

    return 0;
}




