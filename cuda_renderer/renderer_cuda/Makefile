# Executable name
EXEC = kmeans_cuda

# Compilers
NVCC = nvcc
CXX   = g++-11        # Host compiler for nvcc

# --- Compiler Flags ---
NVCCFLAGS = -O3 -std=c++17 -arch=sm_75 -ccbin $(CXX)

# --- Include Directories ---
INCLUDES = -I. -Irenderer_cuda

# --- Source Files ---
CXX_SRCS = main.cpp ../Image.cpp ../Renderer.cpp
CUDA_SRCS = CUDARenderer.cu

# --- Object Files ---
OBJS = $(CXX_SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)

# --- Libraries ---
LIBS = -lglut -lGL -lGLU -lcudart

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean

# Default target
all: $(EXEC)

# Link executable
$(EXEC): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

# Compile .cpp files with nvcc so CUDA headers are visible
%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile .cu files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(EXEC) $(OBJS)