#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define TILE_SIZE 16
#define THREAD_TILE_M 4
#define THREAD_TILE_N 4
#define BM (TILE_SIZE * THREAD_TILE_M)
#define BN (TILE_SIZE * THREAD_TILE_N)

__global__ void gemm_naive_kernel(float* C, const float* A, const float* B,
                                   int M, int N, int K) {
    __shared__ float As[BM][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][BN + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int rowStart = by * BM;
    int colStart = bx * BN;
    
    float accum[THREAD_TILE_M][THREAD_TILE_N];
    
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            accum[i][j] = 0.0f;
        }
    }
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int kStart = t * TILE_SIZE;
        
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            int row = rowStart + ty * THREAD_TILE_M + i;
            int col = kStart + tx;
            if (row < M && col < K) {
                As[ty * THREAD_TILE_M + i][tx] = A[row * K + col];
            } else {
                As[ty * THREAD_TILE_M + i][tx] = 0.0f;
            }
        }
        
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int row = kStart + ty;
            int col = colStart + tx * THREAD_TILE_N + j;
            if (row < K && col < N) {
                Bs[ty][tx * THREAD_TILE_N + j] = B[row * N + col];
            } else {
                Bs[ty][tx * THREAD_TILE_N + j] = 0.0f;
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    accum[i][j] += As[ty * THREAD_TILE_M + i][k] * Bs[k][tx * THREAD_TILE_N + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int row = rowStart + ty * THREAD_TILE_M + i;
            int col = colStart + tx * THREAD_TILE_N + j;
            if (row < M && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}

int main() {
    const int M = 512;
    const int N = 512;
    const int K = 512;
    
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
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
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