#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

__global__ void softmax_naive_kernel(float* output, const float* input,
                                      int batch_size, int feature_dim) {

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int row_offset = batch_idx * feature_dim;

    if (tid == 0) {
        // Pass 1: Find max value sequentially
        float max_val = -1e20f;
        for (int i = 0; i < feature_dim; i++) {
            float val = input[row_offset + i];
            if (val > max_val) max_val = val;
        }
        
        // Pass 2: Compute exp and sum sequentially
        float sum_exp = 0.0f;
        for (int i = 0; i < feature_dim; i++) {
            float exp_val = expf(input[row_offset + i] - max_val);
            output[row_offset + i] = exp_val;
            sum_exp += exp_val;
        }
        
        // Pass 3: Normalize sequentially
        for (int i = 0; i < feature_dim; i++) {
            output[row_offset + i] /= sum_exp;
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
    

    dim3 block(32);
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
