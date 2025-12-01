# # =============================================================================
# # Makefile for a Simple CUDA/OpenGL Renderer
# # =============================================================================

# # Change the CXX compiler
# CXX = g++-11
# NVCC = nvcc

# # ... (rest of the file) ...

# # --- Executable Name ---
# TARGET = renderer

# # --- Source Files ---
# # List all C++ source files
# SOURCES_CPP = main.cpp Image.cpp
# # List all CUDA source files
# SOURCES_CU  = SimpleRenderer.cu

# # --- Object Files ---
# # Automatically generate object file names from the source file lists
# # e.g., "main.cpp Image.cpp" becomes "main.o Image.o"
# OBJECTS = $(SOURCES_CPP:.cpp=.o) $(SOURCES_CU:.cu=.o)

# # --- Compilation Flags ---
# # CXXFLAGS: Flags for the C++ compiler
# # -g          : Generate debugging information
# # -Wall       : Enable all standard compiler warnings
# # -std=c++11  : Use the C++11 standard (for nullptr, etc.)
# # -I.         : Look for header files in the current directory
# CXXFLAGS = -g -Wall -std=c++11 -I.

# # Tell nvcc which host compiler to use via the -ccbin flag
# NVCCFLAGS = -g -std=c++11 -arch=sm_75 -ccbin g++-11


# # --- Linker Flags and Libraries ---
# # LDFLAGS: Flags passed to the linker, like library search paths.
# # Uncomment the line below if your CUDA installation is in a non-standard path.
# # LDFLAGS = -L/usr/local/cuda/lib64

# # LIBS: Libraries to link against.
# # -lglut      : The GLUT library for windowing
# # -lGL        : The core OpenGL library
# # -lcudart    : The CUDA Runtime library (for cudaMalloc, cudaMemcpy, etc.)
# LIBS = -lglut -lGL -lcudart


# # =============================================================================
# # Build Rules
# # =============================================================================

# # --- Phony Targets ---
# # These are targets that don't represent actual files.
# .PHONY: all clean

# # --- Default Target ---
# # The 'all' target is the default. Typing 'make' will build the executable.
# # It depends on the final executable file.
# all: $(TARGET)

# # --- Linking Rule ---
# # This rule links all the compiled object files (.o) into the final executable.
# # We use nvcc for linking because it automatically handles linking the necessary
# # CUDA support libraries.
# # $@ is a special variable for the target name ("renderer").
# # $^ is a special variable for all prerequisites (all the .o files).
# $(TARGET): $(OBJECTS)
# 	$(NVCC) $(LDFLAGS) -o $@ $^ $(LIBS)

# # --- Pattern Rule for C++ Compilation ---
# # This is a generic rule for compiling any .cpp file into a .o file.
# # "To make a .o file, find the corresponding .cpp file..."
# # -c tells the compiler to only compile, not link.
# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# # --- Pattern Rule for CUDA Compilation ---
# # This is a generic rule for compiling any .cu file into a .o file.
# # "To make a .o file, find the corresponding .cu file..."
# # nvcc handles separating and compiling both host and device code.
# %.o: %.cu
# 	$(NVCC) $(NVCCFLAGS) -c $< -o $@


# # --- Clean Rule ---
# # The 'clean' target removes all generated files (the executable and objects).
# # This is useful for forcing a complete rebuild.
# # The '-' before rm tells make to ignore errors if a file doesn't exist.
# clean:
# 	-rm -f $(TARGET) $(OBJECTS)


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