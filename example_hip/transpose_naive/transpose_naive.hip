#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_naive_kernel(float* output, const float* input,
                                        int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile from input with coalesced reads
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < cols && (y + i) < rows) {
            tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * cols + x];
        }
    }
    
    __syncthreads();
    
    // Transpose block coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write tile to output with coalesced writes and transposed read from shared memory
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < rows && (y + i) < cols) {
            output[(y + i) * rows + x] = tile[threadIdx.y + i][threadIdx.x];
        }
    }
}

int main() {
    const int rows = 2048;
    const int cols = 2048;
    
    size_t size = rows * cols * sizeof(float);
    
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    for (int i = 0; i < rows * cols; i++) h_input[i] = (float)(rand() % 100) / 100.0f;
    
    float *d_input, *d_output;
    hipMalloc(&d_input, size);
    hipMalloc(&d_output, size);
    
    hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice);
    
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        transpose_naive_kernel<<<grid, block>>>(d_output, d_input, rows, cols);
    }
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    
    hipMemcpy(h_output, d_output, size, hipMemcpyDeviceToHost);
    
    std::cout << "Perf: " << ms << " ms" << std::endl;
    std::cout << "Result[0]: " << h_output[0] << std::endl;
    
    hipFree(d_input);
    hipFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}