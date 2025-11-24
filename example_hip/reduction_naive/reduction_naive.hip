#include <hip/hip_runtime.h>

__global__ void reduction_naive_kernel(float* output, const float* input, int size) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    __shared__ float shared_data[256];
    
    if (gid < size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();
    
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_data[i];
        }
        atomicAdd(output, sum);
    }
}

