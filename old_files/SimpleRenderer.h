
// Forward declaration to avoid including Image.h in a CUDA header
class Image; 

class SimpleRenderer {
public:
    // Constructor takes the host image to be rendered
    SimpleRenderer(Image* image);
    ~SimpleRenderer();

    // Executes the CUDA kernel to process the image
    void render();

    // Gets the final image after it has been copied back from the GPU
    const Image* getDisplayImage();

    Image* segmentationImage(const Image* inputImage);
    Image* segmentationImageCUDA(const Image* inputImage);

private:
    int width;
    int height;
    
    // Image on the CPU that will be displayed
    Image* displayImage;

    // Pointers to GPU device memory
    float* d_inputImage;
    float* d_outputImage; 
};