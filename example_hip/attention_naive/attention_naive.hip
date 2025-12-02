#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

__global__ void attention_naive_kernel(const float* query, const float* key,
                                        float* attention,
                                        int batch, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * seq_len;
    
    if (idx >= total) return;
    
    // Convert flat index to batch, query position, key position
    int batch_idx = idx / (seq_len * seq_len);
    int remainder = idx % (seq_len * seq_len);
    int q_idx = remainder / seq_len;
    int k_idx = remainder % seq_len;
    
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        int q_pos = (batch_idx * seq_len + q_idx) * head_dim + d;
        int k_pos = (batch_idx * seq_len + k_idx) * head_dim + d;
        score += query[q_pos] * key[k_pos];
    }
    
    // Scale by sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);
    attention[idx] = score * scale;
}

__global__ void attention_softmax_kernel(float* output, const float* value,
                                          const float* attention,
                                          int batch, int seq_len, int head_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len;
    
    if (idx >= total) return;
    
    int batch_idx = idx / seq_len;
    int q_idx = idx % seq_len;
    
    int attn_row = (batch_idx * seq_len + q_idx) * seq_len;
    
    float max_val = -1e20f;
    for (int k = 0; k < seq_len; k++) {
        float val = attention[attn_row + k];
        if (val > max_val) max_val = val;
    }
    
    float sum_exp = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        sum_exp += expf(attention[attn_row + k] - max_val);
    }
    
    // Compute output for each dimension
    for (int d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            float weight = expf(attention[attn_row + k] - max_val) / sum_exp;
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
    
    int total_attn = batch * seq_len * seq_len;
    dim3 block1(64);
    dim3 grid1((total_attn + 63) / 64);
    
    int total_rows = batch * seq_len;
    dim3 block2(32);
    dim3 grid2((total_rows + 31) / 32);
    
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; i++) {
        attention_naive_kernel<<<grid1, block1>>>(d_query, d_key, d_attention, batch, seq_len, head_dim);
        attention_softmax_kernel<<<grid2, block2>>>(d_output, d_value, d_attention, batch, seq_len, head_dim);
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
