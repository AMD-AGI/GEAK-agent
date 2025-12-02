#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>


// Each thread computes one element of C with direct global memory access
__global__ void gemm_naive_kernel(float* C, const float* A, const float* B,
                                   int M, int N, int K) {
   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);
    
    hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    
    dim3 block(8, 8);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; i++) {
        gemm_naive_kernel<<<grid, block>>>(d_C, d_A, d_B, M, N, K);
    }
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    
    hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    
    std::cout << "Perf: " << ms << " ms" << std::endl;
    std::cout << "Result[0]: " << h_C[0] << std::endl;
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
