# Compiler definitions
CXX = g++
NVCC = nvcc
# Target executable name
EXECUTABLE = hpc_kmeans_visualization
# --- Flags ---
# C++ compiler flags: -g for debugging, -Wall for all warnings, -std=c++11 for modern C++
CXXFLAGS = -g -Wall -std=c++11
# NVCC compiler flags: -g -G for CUDA debugging, -std=c++11 to match C++
# Add -arch=sm_XX for your specific GPU architecture for better performance if needed
NVCCFLAGS = -g -G -std=c++11 -ccbin g++-11
# Linker flags: Link against OpenGL and GLUT libraries.
# The CUDA runtime library is added automatically by using nvcc as the linker.
LDFLAGS = -lGL -lglut
# --- Source and Object Files ---
# List all C++ and CUDA source files
CPP_SOURCES = main.cpp Image.cpp
CUDA_SOURCES = HPCRenderer.cu
# Automatically generate object file names (.o) from source file names
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
OBJECTS = $(CPP_OBJECTS) $(CUDA_OBJECTS)
# --- Build Rules ---
# The default target, "all", which builds the executable.
all: $(EXECUTABLE)
# Rule to link all object files into the final executable.
# We use nvcc for linking to automatically include CUDA runtime libraries.
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)
	@echo "Build complete. Run with: ./$(EXECUTABLE) <path_to_image.ppm>"
# Rule to compile a C++ source file (.cpp) into an object file (.o).
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
# Rule to compile a CUDA source file (.cu) into an object file (.o).
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
# --- Utility Rules ---
# Rule to clean up the directory by removing object files and the executable.
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
	@echo "Cleanup complete."