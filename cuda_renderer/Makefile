# # Compiler definitions
# CXX = g++
# NVCC = nvcc
# # Target executable name
# EXECUTABLE = hpc_kmeans_visualization
# # --- Flags ---
# # C++ compiler flags: -g for debugging, -Wall for all warnings, -std=c++11 for modern C++
# CXXFLAGS = -g -Wall -std=c++11
# # NVCC compiler flags: -g -G for CUDA debugging, -std=c++11 to match C++
# # Add -arch=sm_XX for your specific GPU architecture for better performance if needed
# NVCCFLAGS = -g -G -std=c++11 -ccbin g++-11
# # Linker flags: Link against OpenGL and GLUT libraries.
# # The CUDA runtime library is added automatically by using nvcc as the linker.
# LDFLAGS = -lGL -lglut
# # --- Source and Object Files ---
# # List all C++ and CUDA source files
# CPP_SOURCES = main.cpp Image.cpp
# CUDA_SOURCES = HPCRenderer.cu
# # Automatically generate object file names (.o) from source file names
# CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
# CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
# OBJECTS = $(CPP_OBJECTS) $(CUDA_OBJECTS)
# # --- Build Rules ---
# # The default target, "all", which builds the executable.
# all: $(EXECUTABLE)
# # Rule to link all object files into the final executable.
# # We use nvcc for linking to automatically include CUDA runtime libraries.
# $(EXECUTABLE): $(OBJECTS)
# 	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)
# 	@echo "Build complete. Run with: ./$(EXECUTABLE) <path_to_image.ppm>"
# # Rule to compile a C++ source file (.cpp) into an object file (.o).
# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@
# # Rule to compile a CUDA source file (.cu) into an object file (.o).
# %.o: %.cu
# 	$(NVCC) $(NVCCFLAGS) -c $< -o $@
# # --- Utility Rules ---
# # Rule to clean up the directory by removing object files and the executable.
# clean:
# 	rm -f $(OBJECTS) $(EXECUTABLE)
# 	@echo "Cleanup complete."

# Executable name
EXEC = kmeans_renderer

# Compilers
CXX = g++
NVCC = nvcc

# --- Compiler Flags ---
# CXXFLAGS: Flags for the C++ compiler (g++)
# -O3: High optimization level
# -Wall: Enable all warnings
# -std=c++11: Use the C++11 standard (or c++14, c++17)
# -fopenmp: Enable OpenMP support
CXXFLAGS = -O3 -Wall -std=c++11 -fopenmp

# NVCCFLAGS: Flags for the CUDA compiler (nvcc)
# -O3: Optimization level
# -std=c++11: C++ standard for host code
# -arch=sm_75: Target GPU architecture. CHANGE THIS to match your GPU!
#             (e.g., sm_61 for a GTX 1070, sm_86 for an RTX 3080)
# -Xcompiler -fopenmp: Pass the -fopenmp flag to the host compiler (g++)
NVCCFLAGS = -O3 -std=c++11 -arch=sm_75 -Xcompiler -fopenmp

# --- Linker Flags ---
# LDFLAGS: Flags passed to the linker
# -lglut, -lGL, -lGLU: OpenGL libraries
# -lcudart: CUDA runtime library
# -fopenmp: OpenMP library (needed for linking too)
LDFLAGS = -lglut -lGL -lGLU -lcudart -fopenmp

# --- Project Files ---
# Include directories
# We need to tell the compiler where to find header files.
INCLUDES = -I. \
           -Irenderer_cuda \
           -Irenderer_openMP \
           -ISimpleRenderer

# C++ source files
CXX_SRCS = main.cpp \
           Image.cpp

# CUDA source files (.cu)
# NOTE: OpenMPRenderer.cu and HPCRenderer.cu are being compiled with nvcc
# because of their .cu extension. This works, but they could be .cpp files.
CUDA_SRCS = renderer_cuda/CUDARenderer.cu \
            renderer_openMP/OpenMPRenderer.cu \
            SimpleRenderer/HPCRenderer.cu

# Object files are derived automatically from source files
OBJS = $(CXX_SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)

# --- Build Rules ---

# The default target: build the executable
all: $(EXEC)

# Rule to link the final executable
# We use nvcc for linking because it knows how to correctly link CUDA code.
$(EXEC): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

# Rule to compile a .cpp file into a .o file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule to compile a .cu file into a .o file
# This single rule correctly handles files in subdirectories.
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Rule to clean up the project directory
clean:
	rm -f $(EXEC) $(OBJS)

.PHONY: all clean
