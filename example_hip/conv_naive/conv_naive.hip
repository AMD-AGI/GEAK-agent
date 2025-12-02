#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

// Each thread computes one output pixel
__global__ void conv2d_naive_kernel(float* output, const float* input, const float* kernel,
                                     int batch, int in_channels, int out_channels,
                                     int height, int width, int kernel_size,
                                     int out_height, int out_width, int padding, int stride) {
    

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Convert flat index to output coordinates
    int oc = idx / (out_height * out_width);
    int remainder = idx % (out_height * out_width);
    int oh = remainder / out_width;
    int ow = remainder % out_width;
    int b = 0;  // batch = 1 for simplicity
    
    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int ih = oh * stride + ky - padding;
                int iw = ow * stride + kx - padding;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
               
                    int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                   
                    int kernel_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                    
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }
    
    // Write output
    int out_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
    output[out_idx] = sum;
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
    
    int total_output = out_channels * out_height * out_width;
    dim3 block(64);
    dim3 grid((total_output + 63) / 64);
    
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
