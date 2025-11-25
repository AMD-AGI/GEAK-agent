#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define TILE_SIZE 16
#define BLOCK_SIZE 128

__global__ void attention_naive_kernel(const float* query, const float* key,
                                        float* attention,
                                        int batch, int seq_len, int head_dim) {
    int batch_idx = blockIdx.z;
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch || q_idx >= seq_len || k_idx >= seq_len) return;
    
    __shared__ float tile_q[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_k[TILE_SIZE][TILE_SIZE + 1];
    
    float score = 0.0f;
    
    for (int tile = 0; tile < (head_dim + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int d = tile * TILE_SIZE + threadIdx.x;
        if (q_idx < seq_len && d < head_dim) {
            int q_pos = (batch_idx * seq_len + q_idx) * head_dim + d;
            tile_q[threadIdx.y][threadIdx.x] = query[q_pos];
        } else {
            tile_q[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int d2 = tile * TILE_SIZE + threadIdx.y;
        if (k_idx < seq_len && d2 < head_dim) {
            int k_pos = (batch_idx * seq_len + k_idx) * head_dim + d2;
            tile_k[threadIdx.x][threadIdx.y] = key[k_pos];
        } else {
            tile_k[threadIdx.x][threadIdx.y] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            score += tile_q[threadIdx.y][i] * tile_k[threadIdx.x][i];
        }
        
        __syncthreads();
    }
    
    if (q_idx < seq_len && k_idx < seq_len) {
        float scale = 1.0f / sqrtf((float)head_dim);
        int attn_idx = (batch_idx * seq_len + q_idx) * seq_len + k_idx;
        attention[attn_idx] = score * scale;
    }
}

__global__ void attention_softmax_kernel(float* output, const float* value,
                                          const float* attention,
                                          int batch, int seq_len, int head_dim) {
    int q_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch || q_idx >= seq_len) return;
    
    extern __shared__ float shared_mem[];
    float* s_max_vals = shared_mem;
    float* s_sum_vals = &shared_mem[BLOCK_SIZE];
    
    float local_max = -1e20f;
    for (int k = tid; k < seq_len; k += blockDim.x) {
        int attn_idx = (batch_idx * seq_len + q_idx) * seq_len + k;
        local_max = fmaxf(local_max, attention[attn_idx]);
    }
    s_max_vals[tid] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max_vals[tid] = fmaxf(s_max_vals[tid], s_max_vals[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = s_max_vals[0];
    __syncthreads();
    
    float local_sum = 0.0f;
    for (int k = tid; k < seq_len; k += blockDim.x) {
        int attn_idx = (batch_idx * seq_len + q_idx) * seq_len + k;
        float exp_val = expf(attention[attn_idx] - max_val);
        local_sum += exp_val;
    }
    s_sum_vals[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_vals[tid] += s_sum_vals[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = s_sum_vals[0];
    __syncthreads();
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            int attn_idx = (batch_idx * seq_len + q_idx) * seq_len + k;
            float weight = expf(attention[attn_idx] - max_val) / sum_val;
            int v_pos = (batch_idx * seq_len + k) * head_dim + d;
            out_val += weight * value[v_pos];
        }
        int out_idx = (batch_idx * seq_len + q_idx) * head_dim + d;
        output[out_idx] = out_val;
    }
}

int main() {
    const int batch = 2;
    const int seq_len = 128;
    const int head_dim = 64;
    
    size_t qkv_size = batch * seq_len * head_dim * sizeof(float);
    size_t attn_size = batch * seq_len * seq_len * sizeof(float);
    
    float *h_query = (float*)malloc(qkv_size);
    float *h_key = (float*)malloc(qkv_size);
    float *h_value = (float*)malloc(qkv_size);
    float *h_output = (float*)malloc(qkv_size);
    
    for (int i = 0; i < batch * seq_len * head_dim; i++) {
        h_query[i] = (float)(rand() % 100) / 100.0f;
        h_key[i] = (float)(rand() % 100) / 100.0f;
        h_value[i] = (float)(rand() % 100) / 100.0f;
    }
    
    float *d_query, *d_key, *d_value, *d_attention, *d_output;
    hipMalloc(&d_query, qkv_size);
    hipMalloc(&d_key, qkv_size);
    hipMalloc(&d_value, qkv_size);
    hipMalloc(&d_attention, attn_size);
    hipMalloc(&d_output, qkv_size);
    
    hipMemcpy(d_query, h_query, qkv_size, hipMemcpyHostToDevice);
    hipMemcpy(d_key, h_key, qkv_size, hipMemcpyHostToDevice);
    hipMemcpy(d_value, h_value, qkv_size, hipMemcpyHostToDevice);
    
    dim3 block1(TILE_SIZE, TILE_SIZE);
    dim3 grid1((seq_len + TILE_SIZE - 1) / TILE_SIZE, 
               (seq_len + TILE_SIZE - 1) / TILE_SIZE, 
               batch);
    
    dim3 block2(BLOCK_SIZE);
    dim3 grid2(seq_len, batch);
    size_t shared_mem_size = 2 * BLOCK_SIZE * sizeof(float);
    
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; i++) {
        attention_naive_kernel<<<grid1, block1>>>(d_query, d_key, d_attention, batch, seq_len, head_dim);
        attention_softmax_kernel<<<grid2, block2, shared_mem_size>>>(d_output, d_value, d_attention, batch, seq_len, head_dim);
    }
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    
    hipMemcpy(h_output, d_output, qkv_size, hipMemcpyDeviceToHost);
    
    std::cout << "Perf: " << ms << " ms" << std::endl;
    std::cout << "Result[0]: " << h_output[0] << std::endl;
    
    hipFree(d_query);
    hipFree(d_key);
    hipFree(d_value);
    hipFree(d_attention);
    hipFree(d_output);
    free(h_query);
    free(h_key);
    free(h_value);
    free(h_output);
    
    return 0;
}