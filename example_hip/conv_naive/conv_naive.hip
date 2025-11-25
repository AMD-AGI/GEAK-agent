#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#define BLOCK_SIZE 256
#define REG_BLOCK 4
#define MAX_KERNEL_SIZE 11

__global__ void conv2d_naive_kernel(float* output, const float* input, const float* kernel,
                                     int batch, int in_channels, int out_channels,
                                     int height, int width, int kernel_size,
                                     int out_height, int out_width, int padding, int stride) {
    __shared__ float s_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
    
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int total_outputs = batch * out_channels * out_height * out_width;
    int base_idx = block_id * BLOCK_SIZE * REG_BLOCK + tid * REG_BLOCK;
    
    int indices[REG_BLOCK];
    int ow[REG_BLOCK], oh[REG_BLOCK], oc[REG_BLOCK], b[REG_BLOCK];
    float sum[REG_BLOCK];
    
    for (int r = 0; r < REG_BLOCK; r++) {
        indices[r] = base_idx + r;
        sum[r] = 0.0f;
        
        if (indices[r] < total_outputs) {
            ow[r] = indices[r] % out_width;
            oh[r] = (indices[r] / out_width) % out_height;
            oc[r] = (indices[r] / (out_width * out_height)) % out_channels;
            b[r] = indices[r] / (out_width * out_height * out_channels);
        }
    }
    
    for (int ic = 0; ic < in_channels; ic++) {
        int kernel_elements = kernel_size * kernel_size;
        
        for (int r = 0; r < REG_BLOCK; r++) {
            if (indices[r] < total_outputs) {
                int first_oc = oc[r];
                
                for (int i = tid; i < kernel_elements; i += BLOCK_SIZE) {
                    int kernel_idx = ((first_oc * in_channels + ic) * kernel_size * kernel_size) + i;
                    s_kernel[i] = kernel[kernel_idx];
                }
                __syncthreads();
                
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int ih = oh[r] * stride + ky - padding;
                        int iw = ow[r] * stride + kx - padding;
                        
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int input_idx = ((b[r] * in_channels + ic) * height + ih) * width + iw;
                            int k_idx = ky * kernel_size + kx;
                            sum[r] += input[input_idx] * s_kernel[k_idx];
                        }
                    }
                }
                __syncthreads();
                break;
            }
        }
    }
    
    for (int r = 0; r < REG_BLOCK; r++) {
        if (indices[r] < total_outputs) {
            output[indices[r]] = sum[r];
        }
    }
}

int main() {
    const int batch = 1;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 64;
    const int width = 64;
    const int kernel_size = 3;
    const int padding = 1;
    const int stride = 1;
    const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    size_t input_size = batch * in_channels * height * width * sizeof(float);
    size_t kernel_size_bytes = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = batch * out_channels * out_height * out_width * sizeof(float);
    
    float *h_input = (float*)malloc(input_size);
    float *h_kernel = (float*)malloc(kernel_size_bytes);
    float *h_output = (float*)malloc(output_size);
    
    for (int i = 0; i < batch * in_channels * height * width; i++) 
        h_input[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size; i++) 
        h_kernel[i] = (float)(rand() % 100) / 100.0f;
    
    float *d_input, *d_kernel, *d_output;
    hipMalloc(&d_input, input_size);
    hipMalloc(&d_kernel, kernel_size_bytes);
    hipMalloc(&d_output, output_size);
    
    hipMemcpy(d_input, h_input, input_size, hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, h_kernel, kernel_size_bytes, hipMemcpyHostToDevice);
    
    int total = batch * out_channels * out_height * out_width;
    dim3 block(BLOCK_SIZE);
    dim3 grid((total + BLOCK_SIZE * REG_BLOCK - 1) / (BLOCK_SIZE * REG_BLOCK));
    
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        conv2d_naive_kernel<<<grid, block>>>(d_output, d_input, d_kernel,
            batch, in_channels, out_channels, height, width, kernel_size,
            out_height, out_width, padding, stride);
    }
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    
    hipMemcpy(h_output, d_output, output_size, hipMemcpyDeviceToHost);
    
    std::cout << "Perf: " << ms << " ms" << std::endl;
    std::cout << "Result[0]: " << h_output[0] << std::endl;
    
    hipFree(d_input);
    hipFree(d_kernel);
    hipFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output);
    
    return 0;
}