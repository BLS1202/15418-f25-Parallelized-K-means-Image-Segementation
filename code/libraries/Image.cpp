#include "Image.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

bool Image::loadJPG(const std::string& filename) {

    int w, h, channels_in_file;

    unsigned char *ppm_data = stbi_load(
        filename.c_str(), 
        &w, 
        &h, 
        &channels_in_file, 
        4 // Force loading as 4 channels (RGBA)
    );

    if (ppm_data == nullptr) {
        std::cerr << "Error: Could not load JPG/Image file '" << filename << "'." << std::endl;
        std::cerr << "Reason: " << stbi_failure_reason() << std::endl;
        return false;
    }

    // Free the old memory and update dimensions
    if (data) {
        delete[] data;
    }
    width = w;
    height = h;
    
    // Allocate new memory for our 32-bit float RGBA data
    data = new float[width * height * 4];


    for (int i = 0; i < width * height; ++i) {
        // stbi_load gives us R, G, B, A in 8-bit format
        data[i * 4 + 0] = static_cast<float>(ppm_data[i * 4 + 0]) / 255.0f; // R
        data[i * 4 + 1] = static_cast<float>(ppm_data[i * 4 + 1]) / 255.0f; // G
        data[i * 4 + 2] = static_cast<float>(ppm_data[i * 4 + 2]) / 255.0f; // B
        data[i * 4 + 3] = static_cast<float>(ppm_data[i * 4 + 3]) / 255.0f; // A
    }


    stbi_image_free(ppm_data); 
    
    std::cout << "Loaded image '" << filename << "' (" << width << "x" << height << ") as 32-bit RGBA." << std::endl;
    return true;
}

bool Image::save_image_to_jpg(const std::string& filename, const std::vector<unsigned char>& image_data, int width, int height, int quality=90) {
    int success = stbi_write_jpg(
        filename.c_str(), 
        width, 
        height, 
        3, // Number of channels (RGB)
        image_data.data(),
        quality
    );

    if (success) {
        std::cout << "\nSuccessfully saved image to '" << filename << "'" << std::endl;
        return true;
    } else {
        std::cerr << "\nError: Failed to write image to '" << filename << "'" << std::endl;
        return false;
    }
}