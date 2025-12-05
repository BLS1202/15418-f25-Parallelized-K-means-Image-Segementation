#include "Image.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

Image::Image(int w, int h) : width(w), height(h) {
    data = new float[width * height * 4]; // 4 channels: R, G, B, A
}

Image::~Image() {
    delete[] data;
}

bool Image::loadPPM(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        return false;
    }

    std::string magic;
    ifs >> magic;

    if (magic != "P6") {
        std::cerr << "Error: Invalid PPM file. Must be P6 (binary)." << std::endl;
        return false;
    }

    ifs >> width >> height;
    int max_val;
    ifs >> max_val;
    ifs.get(); // Consume the single whitespace character

    if (max_val != 255) {
        std::cerr << "Error: PPM max color value must be 255." << std::endl;
        return false;
    }
    
    delete[] data;
    data = new float[width * height * 4];

    std::vector<unsigned char> ppm_data(width * height * 3);
    ifs.read(reinterpret_cast<char*>(ppm_data.data()), ppm_data.size());

    if (!ifs) {
        std::cerr << "Error: Failed to read pixel data from " << filename << std::endl;
        return false;
    }

    // Convert 8-bit RGB to 32-bit float RGBA
    for (int i = 0; i < width * height; ++i) {
        data[i * 4 + 0] = static_cast<float>(ppm_data[i * 3 + 0]) / 255.0f; // R
        data[i * 4 + 1] = static_cast<float>(ppm_data[i * 3 + 1]) / 255.0f; // G
        data[i * 4 + 2] = static_cast<float>(ppm_data[i * 3 + 2]) / 255.0f; // B
        data[i * 4 + 3] = 1.0f;                                            // A
    }
    
    std::cout << "Loaded image '" << filename << "' (" << width << "x" << height << ")" << std::endl;
    return true;
}
