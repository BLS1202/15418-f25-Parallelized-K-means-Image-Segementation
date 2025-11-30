#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// The two crucial lines for using the stb_image_write library
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[]) {
    // Check if the user provided the input and output filenames
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.ppm> <output.jpg_or_png>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];
    std::string output_filename = argv[2];

    // --- Step 1: Read the PPM file ---
    std::ifstream ppm_file(input_filename, std::ios::in | std::ios::binary);
    if (!ppm_file) {
        std::cerr << "Error: Could not open PPM file '" << input_filename << "'" << std::endl;
        return 1;
    }

    std::string magic_number;
    int width, height, max_val;

    // Read the header
    ppm_file >> magic_number;
    if (magic_number != "P6") {
        std::cerr << "Error: Input file is not a binary P6 PPM." << std::endl;
        return 1;
    }
    
    // Skip comments in the header
    while (ppm_file.peek() == '\n' || ppm_file.peek() == ' ') { ppm_file.get(); }
    while (ppm_file.peek() == '#') {
        ppm_file.ignore(256, '\n');
    }

    ppm_file >> width >> height >> max_val;
    
    // The next character should be a newline or space
    ppm_file.ignore(1, '\n');

    std::cout << "Reading PPM: " << width << "x" << height << std::endl;

    // Read the raw binary pixel data
    std::vector<unsigned char> data(width * height * 3);
    ppm_file.read(reinterpret_cast<char*>(data.data()), data.size());
    ppm_file.close();


    // --- Step 2: Write the JPG/PNG file using stb_image_write ---
    
    // Determine the output format from the filename extension
    std::string extension = output_filename.substr(output_filename.find_last_of(".") + 1);
    bool success = false;

    if (extension == "jpg" || extension == "jpeg") {
        // Write a JPG file. The last parameter is quality (1-100).
        int quality = 95;
        success = stbi_write_jpg(output_filename.c_str(), width, height, 3, data.data(), quality);
    } else if (extension == "png") {
        // Write a PNG file. The "stride" is the number of bytes per row.
        int stride_in_bytes = width * 3;
        success = stbi_write_png(output_filename.c_str(), width, height, 3, data.data(), stride_in_bytes);
    } else if (extension == "bmp") {
        // Write a BMP file.
        success = stbi_write_bmp(output_filename.c_str(), width, height, 3, data.data());
    } else {
        std::cerr << "Error: Unsupported output format '" << extension << "'. Please use jpg, png, or bmp." << std::endl;
        return 1;
    }

    if (success) {
        std::cout << "Successfully saved image to '" << output_filename << "'" << std::endl;
    } else {
        std::cerr << "Error: Failed to write image to '" << output_filename << "'" << std::endl;
        return 1;
    }

    return 0;
}
