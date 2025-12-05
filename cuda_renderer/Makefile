
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc

# C++ compiler flags
CXXFLAGS := -I. -O2 -Wall

# CUDA compiler flags
# -dc is for relocatable device code, necessary for linking multiple .cu files
NVCCFLAGS := -I. -O2 -g -G -std=c++11 -ccbin g++-11

# Linker flags
# Add CUDA runtime library and any other libraries (like OpenGL, GLUT)
LDFLAGS := -L/usr/local/cuda/lib64 -lcudart -lglut -lGLU -lGL


TARGET := kmeans_visualizer

# Directories
SRC_DIR_SIMPLE := SimpleRenderer
SRC_DIR_OMP := renderer_openMP
SRC_DIR_CUDA := renderer_cuda

# Source files
SRCS_CXX := main.cpp Image.cpp Renderer.cpp
SRCS_NVCC := $(SRC_DIR_SIMPLE)/HPCRenderer.cu \
             $(SRC_DIR_OMP)/OpenMPrenderer.cu \
             $(SRC_DIR_CUDA)/CUDARenderer.cu

# Object files (derived from source files)
OBJS_CXX := $(SRCS_CXX:.cpp=.o)


OBJS_NVCC := $(notdir $(SRCS_NVCC:.cu=.o))

# All object files
OBJS := $(OBJS_CXX) $(OBJS_NVCC)


all: $(TARGET)

# Linking rule for the final executable
$(TARGET): $(OBJS)
	@echo "--- Linking all object files ---"
	# The final link step must use nvcc to handle device code
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)



# Rule to compile a .cpp file into a .o file
%.o: %.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile a .cu file from the SimpleRenderer directory
%.o: $(SRC_DIR_SIMPLE)/%.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to compile a .cu file from the renderer_openMP directory
%.o: $(SRC_DIR_OMP)/%.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to compile a .cu file from the renderer_cuda directory
%.o: $(SRC_DIR_CUDA)/%.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


clean:
	@echo "--- Cleaning up project files ---"
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean




