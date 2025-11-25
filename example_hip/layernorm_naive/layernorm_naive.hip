#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 64);
    }
    return val;
}

__global__ void layernorm_naive_kernel(float* output, const float* input,
                                        const float* gamma, const float* beta,
                                        int batch_size, int feature_dim, float epsilon) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 63;
    int warp_id = tid >> 6;
    
    if (batch_idx >= batch_size) return;
    
    int row_offset = batch_idx * feature_dim;
    
    const int num_warps = (blockDim.x + 63) >> 6;
    __shared__ float shared_data[8];
    
    // Fused pass: compute sum and sum of squares
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float val = input[row_offset + i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }
    
    // Warp-level reduction for sum
    float warp_sum = warpReduceSum(thread_sum);
    float warp_sum_sq = warpReduceSum(thread_sum_sq);
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared_data[warp_id] = warp_sum;
        shared_data[warp_id + 4] = warp_sum_sq;
    }
    __syncthreads();
    
    // Final reduction in first warp
    float mean, inv_std;
    if (tid < num_warps) {
        float val_sum = (tid < num_warps) ? shared_data[tid] : 0.0f;
        float val_sum_sq = (tid < num_warps) ? shared_data[tid + 4] : 0.0f;
        val_sum = warpReduceSum(val_sum);
        val_sum_sq = warpReduceSum(val_sum_sq);
        if (tid == 0) {
            mean = val_sum / feature_dim;
            float variance = (val_sum_sq / feature_dim) - (mean * mean);
            inv_std = rsqrtf(variance + epsilon);
            shared_data[0] = mean;
            shared_data[1] = inv_std;
        }
    }
    __syncthreads();
    mean = shared_data[0];
    inv_std = shared_data[1];
    
    // Normalize and apply affine transformation
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float normalized = (input[row_offset + i] - mean) * inv_std;
        output[row_offset + i] = gamma[i] * normalized + beta[i];
    }
}

int main() {
    const int batch_size = 32;
    const int feature_dim = 768;
    const float epsilon = 1e-5f;
    
    size_t data_size = batch_size * feature_dim * sizeof(float);
    size_t param_size = feature_dim * sizeof(float);
    
    float *h_input = (float*)malloc(data_size);
    float *h_output = (float*)malloc(data_size);
    float *h_gamma = (float*)malloc(param_size);
    float *h_beta = (float*)malloc(param_size);
    
    for (int i = 0; i < batch_size * feature_dim; i++) 
        h_input[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    for (int i = 0; i < feature_dim; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    
    float *d_input, *d_output, *d_gamma, *d_beta;
    hipMalloc(&d_input, data_size);
    hipMalloc(&d_output, data_size);
    hipMalloc(&d_gamma, param_size);
    hipMalloc(&d_beta, param_size);
    
    hipMemcpy(d_input, h_input, data_size, hipMemcpyHostToDevice);
    hipMemcpy(d_gamma, h_gamma, param_size, hipMemcpyHostToDevice);
    hipMemcpy(d_beta, h_beta, param_size, hipMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid(batch_size);
    
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        layernorm_naive_kernel<<<grid, block>>>(d_output, d_input, d_gamma, d_beta,
                                                 batch_size, feature_dim, epsilon);
    }
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    
    hipMemcpy(h_output, d_output, data_size, hipMemcpyDeviceToHost);
    
    std::cout << "Perf: " << ms << " ms" << std::endl;
    std::cout << "Result[0]: " << h_output[0] << std::endl;
    
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_gamma);
    hipFree(d_beta);
    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);
    
    return 0;
}