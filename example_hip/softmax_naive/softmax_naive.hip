#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define WARP_SIZE 64
#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void softmax_naive_kernel(float* output, const float* input,
                                      int batch_size, int feature_dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    int row_offset = batch_idx * feature_dim;
    
    __shared__ float warp_maxs[NUM_WARPS];
    __shared__ float warp_sums[NUM_WARPS];
    
    // Phase 1: Find max value
    float my_max = -1e20f;
    
    // Vectorized loads if feature_dim is divisible by 4
    if (feature_dim % 4 == 0) {
        int num_vec = feature_dim / 4;
        for (int i = tid; i < num_vec; i += BLOCK_SIZE) {
            float4 vals = reinterpret_cast<const float4*>(input + row_offset)[i];
            my_max = fmaxf(my_max, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
        }
    } else {
        for (int i = tid; i < feature_dim; i += BLOCK_SIZE) {
            my_max = fmaxf(my_max, input[row_offset + i]);
        }
    }
    
    // Warp-level max reduction
    float warp_max = warpReduceMax(my_max);
    if (lane == 0) {
        warp_maxs[warp_id] = warp_max;
    }
    __syncthreads();
    
    // Cross-warp max reduction
    float max_val;
    if (warp_id == 0) {
        float val = (lane < NUM_WARPS) ? warp_maxs[lane] : -1e20f;
        val = warpReduceMax(val);
        if (lane == 0) {
            warp_maxs[0] = val;
        }
    }
    __syncthreads();
    max_val = warp_maxs[0];
    
    // Phase 2: Compute exp and sum
    float my_sum = 0.0f;
    
    if (feature_dim % 4 == 0) {
        int num_vec = feature_dim / 4;
        for (int i = tid; i < num_vec; i += BLOCK_SIZE) {
            float4 vals = reinterpret_cast<const float4*>(input + row_offset)[i];
            float4 exp_vals;
            exp_vals.x = expf(vals.x - max_val);
            exp_vals.y = expf(vals.y - max_val);
            exp_vals.z = expf(vals.z - max_val);
            exp_vals.w = expf(vals.w - max_val);
            reinterpret_cast<float4*>(output + row_offset)[i] = exp_vals;
            my_sum += exp_vals.x + exp_vals.y + exp_vals.z + exp_vals.w;
        }
    } else {
        for (int i = tid; i < feature_dim; i += BLOCK_SIZE) {
            float exp_val = expf(input[row_offset + i] - max_val);
            output[row_offset + i] = exp_val;
            my_sum += exp_val;
        }
    }
    
    // Warp-level sum reduction
    float warp_sum = warpReduceSum(my_sum);
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Cross-warp sum reduction
    float sum_exp;
    if (warp_id == 0) {
        float val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        val = warpReduceSum(val);
        if (lane == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();
    sum_exp = warp_sums[0];
    
    // Phase 3: Normalize
    float inv_sum = 1.0f / sum_exp;
    
    if (feature_dim % 4 == 0) {
        int num_vec = feature_dim / 4;
        for (int i = tid; i < num_vec; i += BLOCK_SIZE) {
            float4 vals = reinterpret_cast<float4*>(output + row_offset)[i];
            vals.x *= inv_sum;
            vals.y *= inv_sum;
            vals.z *= inv_sum;
            vals.w *= inv_sum;
            reinterpret_cast<float4*>(output + row_offset)[i] = vals;
        }
    } else {
        for (int i = tid; i < feature_dim; i += BLOCK_SIZE) {
            output[row_offset + i] *= inv_sum;
        }
    }
}

int main() {
    const int batch_size = 32;
    const int feature_dim = 1024;
    
    size_t data_size = batch_size * feature_dim * sizeof(float);
    
    float *h_input = (float*)malloc(data_size);
    float *h_output = (float*)malloc(data_size);
    
    for (int i = 0; i < batch_size * feature_dim; i++) 
        h_input[i] = (float)(rand() % 100) / 100.0f;
    
    float *d_input, *d_output;
    hipMalloc(&d_input, data_size);
    hipMalloc(&d_output, data_size);
    
    hipMemcpy(d_input, h_input, data_size, hipMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(batch_size);
    
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        softmax_naive_kernel<<<grid, block>>>(d_output, d_input, batch_size, feature_dim);
    }
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    
    hipMemcpy(h_output, d_output, data_size, hipMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < feature_dim; i++) sum += h_output[i];
    
    std::cout << "Perf: " << ms << " ms" << std::endl;
    std::cout << "Result[0]: " << h_output[0] << " (row sum: " << sum << ", expected ~1.0)" << std::endl;
    
    hipFree(d_input);
    hipFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}