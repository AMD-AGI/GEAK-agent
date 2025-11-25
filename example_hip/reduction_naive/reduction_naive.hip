#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void reduction_naive_kernel(float* output, const float* input, int size) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + tid;
    
    __shared__ float shared_data[BLOCK_SIZE];
    
    // Each thread loads and sums multiple elements
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = gid + i * blockDim.x;
        if (idx < size) {
            sum += input[idx];
        }
    }
    
    shared_data[tid] = sum;
    __syncthreads();
    
    // Parallel tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > warpSize; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Final reduction at warp level
    if (tid < warpSize) {
        float val = shared_data[tid];
        // Accumulate remaining elements into first warp
        for (int i = warpSize; i < blockDim.x; i += warpSize) {
            if (tid + i < blockDim.x) {
                val += shared_data[tid + i];
            }
        }
        // Warp-level reduction using shuffle intrinsics
        val = warpReduceSum(val);
        
        if (tid == 0) {
            atomicAdd(output, val);
        }
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
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((size + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD));
    
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