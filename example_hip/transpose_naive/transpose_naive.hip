#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

__global__ void transpose_naive_kernel(float* output, const float* input,
                                        int rows, int cols) {
    // Simple 1D flattened indexing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        // Convert linear index to 2D coordinates
        int row = idx / cols;
        int col = idx % cols;
        
       
        output[col * rows + row] = input[row * cols + col];
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
    

    int total_elements = rows * cols;
    dim3 block(64);
    dim3 grid((total_elements + block.x - 1) / block.x);
    
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
