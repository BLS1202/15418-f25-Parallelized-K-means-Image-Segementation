#include <iostream>
#include <fstream>
#include <vector>
#include <string>
// The two crucial lines for using the stb_image library
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// A helper function to write the image data to a P6 PPM file
void write_ppm(const std::string& filename, int width, int height, unsigned char* data) {
    // Open the file for writing in binary mode
    std::ofstream ppm_file(filename, std::ios::out | std::ios::binary);
    if (!ppm_file) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    // Write the PPM header
    // P6 is the magic number for a binary PPM file
    // Width and height are separated by a space
    // 255 is the maximum color value
    ppm_file << "P6\n";
    ppm_file << width << " " << height << "\n";
    ppm_file << "255\n";
    // Write the raw pixel data
    // The image has 'width * height' pixels, and each pixel has 3 bytes (R, G, B)
    ppm_file.write(reinterpret_cast<const char*>(data), width * height * 3);
    ppm_file.close();
    std::cout << "Successfully converted image and saved to '" << filename << "'" << std::endl;
}
int main(int argc, char* argv[]) {
    // Check if the user provided the input and output filenames
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.jpg> <output.ppm>" << std::endl;
        return 1;
    }
    std::string input_filename = argv[1];
    std::string output_filename = argv[2];
    // Variables to store image properties
    int width, height, channels;
    // Use stb_image to load the JPG file
    // The last argument '3' forces the image to be loaded with 3 channels (RGB),
    // which is perfect for our PPM output. It will handle JPGs that have
    // an alpha channel (4 channels) or are grayscale (1 channel).
    unsigned char *img_data = stbi_load(input_filename.c_str(), &width, &height, &channels, 3);
    // Check if the image was loaded successfully
    if (img_data == nullptr) {
        std::cerr << "Error: Could not load image '" << input_filename << "'." << std::endl;
        std::cerr << "Reason: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << width << "x" << height << ", with " << channels << " channels." << std::endl;
    std::cout << "Forcing to 3 channels for PPM." << std::endl;
    
    // Call our function to write the raw data into the PPM format
    write_ppm(output_filename, width, height, img_data);
    // IMPORTANT: Free the memory allocated by stb_image
    stbi_image_free(img_data);
    return 0;
}