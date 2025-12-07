#include <string>
#include <vector>

class Image {
public:
    int width;
    int height;
    // We use float data for compatibility with OpenGL's glDrawPixels
    float* data;

    Image(int w, int h);
    ~Image();

    // Load a PPM image from a file
    bool loadPPM(const std::string& filename);
    bool loadJPG(const std::string& filename);
    bool save_image_to_jpg(const std::string& filename, const std::vector<unsigned char>& image_data, int width, int height, int quality);

private:
    // Disable copy constructor and assignment operator
    Image(const Image&);
    Image& operator=(const Image&);
};
