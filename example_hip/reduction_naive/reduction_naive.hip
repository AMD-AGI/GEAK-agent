#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

__global__ void reduction_naive_kernel(float* output, const float* input, int size) {
    __shared__ float shared_data[32];  // Small shared memory for 32 threads
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Each thread loads one element
    if (gid < size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();
    
    
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_data[i];
        }
        // Use atomicAdd to accumulate across blocks (slow!)
        atomicAdd(output, sum);
    }
}

int main() {
    const int size = 1024 * 1024;
    size_t bytes = size * sizeof(float);
    
    float *h_input = (float*)malloc(bytes);
    float h_output = 0.0f;
    
    for (int i = 0; i < size; i++) h_input[i] = 1.0f;
    
    float *d_input, *d_output;
    hipMalloc(&d_input, bytes);
    hipMalloc(&d_output, sizeof(float));
    
    hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice);
    
    dim3 block(32);
    dim3 grid((size + block.x - 1) / block.x);
    
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        hipMemset(d_output, 0, sizeof(float));
        reduction_naive_kernel<<<grid, block>>>(d_output, d_input, size);
    }
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    
    hipMemcpy(&h_output, d_output, sizeof(float), hipMemcpyDeviceToHost);
    
    std::cout << "Perf: " << ms << " ms" << std::endl;
    std::cout << "Result: " << h_output << " (expected: " << size << ")" << std::endl;
    
    hipFree(d_input);
    hipFree(d_output);
    free(h_input);
    
    return 0;
}
