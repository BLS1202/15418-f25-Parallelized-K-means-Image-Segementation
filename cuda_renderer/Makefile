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
PROG = kmeans

# Compilers
CXX = g++
NVCC = nvcc

# --- Source Files ---
# List all C++ source files (to be compiled with g++)
SRCS_CPP = \
    main.cpp \
    lib/image.cpp \
    lib/lodepng.cpp \
    Renderer.cpp \
    renderer_hpc/HPCRenderer.cpp \
    renderer_openmp/OpenMPRenderer.cpp

# List all CUDA source files (to be compiled with nvcc)
# Note: We list the .cpp file, but it will be compiled with nvcc
SRCS_CU = \
    renderer_cuda/CudaRenderer.cpp

# Generate object file names from source files
OBJS = $(SRCS_CPP:.cpp=.o) $(SRCS_CU:.cpp=.o)

# --- Compiler and Linker Flags ---

# Include paths for all compilers
INCLUDES = \
    -Ilib \
    -Irenderer_hpc \
    -Irenderer_openmp \
    -Irenderer_cuda

# C++ compiler flags
# -O3 for optimization, -Wall for warnings, -fopenmp for OpenMP
CXXFLAGS = -O3 -Wall -std=c++11 -fopenmp $(INCLUDES)

# CUDA compiler flags
# TODO: Change -arch=sm_XX to match your GPU's compute capability if needed
NVCCFLAGS = -O3 -arch=sm_35 -rdc=true --compiler-options -fPIC $(INCLUDES)

# Linker flags
# -fopenmp is needed here as well for linking
LDFLAGS = -fopenmp

# Libraries to link against (OpenGL, CUDA runtime)
LIBS = -lglut -lGL -lGLEW -lcudart


# --- Build Rules ---

# Default target: build the executable
all: $(PROG)

# Rule to link the final executable
$(PROG): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(PROG) $(OBJS) $(LIBS)

# Rule to compile a C++ source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Special rule to compile the CUDA C++ source file with nvcc
# This pattern matches the specific CudaRenderer source file
renderer_cuda/CudaRenderer.o: renderer_cuda/CudaRenderer.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --- Dependency Generation (Advanced but recommended) ---
# This automatically figures out which .h files are needed by each .cpp file

# Generate dependency files (.d)
DEPFLAGS = -MMD -MP

# Include dependency files if they exist
-include $(OBJS:.o=.d)

# Modify the compile rules to generate dependency files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

renderer_cuda/CudaRenderer.o: renderer_cuda/CudaRenderer.cpp
	$(NVCC) $(NVCCFLAGS) --compiler-options "$(DEPFLAGS)" -c $< -o $@


# --- Cleanup ---

# Rule to remove generated files
clean:
	rm -f $(PROG) $(OBJS)
	rm -f $(OBJS:.o=.d) # Remove dependency files as well

.PHONY: all clean
