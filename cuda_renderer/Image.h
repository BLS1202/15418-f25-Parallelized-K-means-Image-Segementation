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

private:
    // Disable copy constructor and assignment operator
    Image(const Image&);
    Image& operator=(const Image&);
};
