#include <hip/hip_runtime.h>

__global__ void softmax_naive_kernel(float* output, const float* input,
                                      int batch_size, int feature_dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int row_offset = batch_idx * feature_dim;
    
    __shared__ float max_vals[256];
    float my_max = -1e20f;
    
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float val = input[row_offset + i];
        if (val > my_max) my_max = val;
    }
    max_vals[tid] = my_max;
    __syncthreads();
    
    if (tid == 0) {
        float global_max = max_vals[0];
        for (int i = 1; i < blockDim.x && i < 256; i++) {
            if (max_vals[i] > global_max) {
                global_max = max_vals[i];
            }
        }
        max_vals[0] = global_max;
    }
    __syncthreads();
    float max_val = max_vals[0];
    
    __shared__ float sum_vals[256];
    float my_sum = 0.0f;
    
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float exp_val = expf(input[row_offset + i] - max_val);
        output[row_offset + i] = exp_val;
        my_sum += exp_val;
    }
    sum_vals[tid] = my_sum;
    __syncthreads();
    
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < blockDim.x && i < 256; i++) {
            total_sum += sum_vals[i];
        }
        sum_vals[0] = total_sum;
    }
    __syncthreads();
    float sum_exp = sum_vals[0];
    
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        output[row_offset + i] = output[row_offset + i] / sum_exp;
    }
}

