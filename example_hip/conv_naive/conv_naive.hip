#include <hip/hip_runtime.h>

__global__ void conv2d_naive_kernel(float* output, const float* input, const float* kernel,
                                     int batch, int in_channels, int out_channels,
                                     int height, int width, int kernel_size,
                                     int out_height, int out_width, int padding, int stride) {
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;
    
    if (b >= batch || oc >= out_channels || oh >= out_height || ow >= out_width) return;
    
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
    
    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
    output[output_idx] = sum;
}

