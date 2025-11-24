#include <hip/hip_runtime.h>

__global__ void layernorm_naive_kernel(float* output, const float* input,
                                        const float* gamma, const float* beta,
                                        int batch_size, int feature_dim, float epsilon) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int row_offset = batch_idx * feature_dim;
    
    __shared__ float mean_vals[256];
    __shared__ float var_vals[256];
    
    float my_sum = 0.0f;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        my_sum += input[row_offset + i];
    }
    mean_vals[tid] = my_sum;
    __syncthreads();
    
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x && i < 256; i++) {
            total += mean_vals[i];
        }
        mean_vals[0] = total / feature_dim;
    }
    __syncthreads();
    float mean = mean_vals[0];
    
    float my_var = 0.0f;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float diff = input[row_offset + i] - mean;
        my_var += diff * diff;
    }
    var_vals[tid] = my_var;
    __syncthreads();
    
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x && i < 256; i++) {
            total += var_vals[i];
        }
        var_vals[0] = total / feature_dim;
    }
    __syncthreads();
    float variance = var_vals[0];
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float normalized = (input[row_offset + i] - mean) * inv_std;
        output[row_offset + i] = gamma[i] * normalized + beta[i];
    }
}

