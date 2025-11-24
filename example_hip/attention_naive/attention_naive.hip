#include <hip/hip_runtime.h>

__global__ void attention_naive_kernel(const float* query, const float* key,
                                        float* attention,
                                        int batch, int seq_len, int head_dim) {
    int batch_idx = blockIdx.z;
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch || q_idx >= seq_len || k_idx >= seq_len) return;
    
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        int q_pos = (batch_idx * seq_len + q_idx) * head_dim + d;
        int k_pos = (batch_idx * seq_len + k_idx) * head_dim + d;
        score += query[q_pos] * key[k_pos];
    }
    
    float scale = 1.0f / sqrtf((float)head_dim);
    score = score * scale;
    
    int attn_idx = (batch_idx * seq_len + q_idx) * seq_len + k_idx;
    attention[attn_idx] = score;
}

__global__ void attention_softmax_kernel(float* output, const float* value,
                                          const float* attention,
                                          int batch, int seq_len, int head_dim) {
    int batch_idx = blockIdx.z;
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int head_d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch || q_idx >= seq_len || head_d >= head_dim) return;
    
    float max_score = -1e20f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        int attn_idx = (batch_idx * seq_len + q_idx) * seq_len + k_idx;
        float score = attention[attn_idx];
        if (score > max_score) max_score = score;
    }
    
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        int attn_idx = (batch_idx * seq_len + q_idx) * seq_len + k_idx;
        sum_exp += expf(attention[attn_idx] - max_score);
    }
    
    float out_val = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        int attn_idx = (batch_idx * seq_len + q_idx) * seq_len + k_idx;
        float weight = expf(attention[attn_idx] - max_score) / sum_exp;
        int v_pos = (batch_idx * seq_len + k_idx) * head_dim + head_d;
        out_val += weight * value[v_pos];
    }
    
    int output_idx = (batch_idx * seq_len + q_idx) * head_dim + head_d;
    output[output_idx] = out_val;
}

