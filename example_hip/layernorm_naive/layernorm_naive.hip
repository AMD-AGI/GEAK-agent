#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define WARP_SIZE 64

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, WARP_SIZE);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[4];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        val = shared[lane];
    } else {
        val = 0.0f;
    }
    
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__global__ void layernorm_naive_kernel(float* output, const float* input,
                                        const float* gamma, const float* beta,
                                        int batch_size, int feature_dim, float epsilon) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int row_offset = batch_idx * feature_dim;
    
    // Vectorized pointers
    const float4* input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4* output4 = reinterpret_cast<float4*>(output + row_offset);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4 = reinterpret_cast<const float4*>(beta);
    int feature_dim4 = feature_dim / 4;
    
    // Fused pass: Compute sum and sum of squares with vectorized loads
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    
    #pragma unroll 2
    for (int i = tid; i < feature_dim4; i += blockDim.x) {
        float4 val = input4[i];
        thread_sum += val.x + val.y + val.z + val.w;
        thread_sq_sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    
    // Reduce sum
    float total_sum = blockReduceSum(thread_sum);
    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = total_sum / feature_dim;
    }
    __syncthreads();
    float mean = s_mean;
    
    // Reduce sum of squares
    float total_sq_sum = blockReduceSum(thread_sq_sum);
    __shared__ float s_inv_std;
    if (tid == 0) {
        float variance = (total_sq_sum / feature_dim) - (mean * mean);
        s_inv_std = rsqrtf(variance + epsilon);
    }
    __syncthreads();
    float inv_std = s_inv_std;
    
    // Normalize and apply affine transformation with vectorized operations
    #pragma unroll 2
    for (int i = tid; i < feature_dim4; i += blockDim.x) {
        float4 val = input4[i];
        float4 g = gamma4[i];
        float4 b = beta4[i];
        
        float4 result;
        result.x = g.x * ((val.x - mean) * inv_std) + b.x;
        result.y = g.y * ((val.y - mean) * inv_std) + b.y;
        result.z = g.z * ((val.z - mean) * inv_std) + b.z;
        result.w = g.w * ((val.w - mean) * inv_std) + b.w;
        
        output4[i] = result;
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