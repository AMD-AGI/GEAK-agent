#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  std::exit(1);} } while(0)

using bf16 = __hip_bfloat16;

__device__ __forceinline__ float silu_f(float x){
  float neg_x = -x;
  float exp_neg_x = expf(neg_x);
  float sigmoid = 1.0f / (1.0f + exp_neg_x);
  return x * sigmoid;
}

__global__ void silu_mul_kernel(
    bf16* __restrict__ out,
    const bf16* __restrict__ in,
    int64_t B, int64_t H)
{
  int64_t b = blockIdx.x;
  
  if (b >= B) return;
  
  const int64_t x_base = b * 2 * H;
  const int64_t y_base = b * 2 * H + H;
  const int64_t out_base = b * H;
  
  int64_t tid = threadIdx.x;
  int64_t stride = blockDim.x * 4;
  
  // Main loop: process 4 elements per thread per iteration
  for (int64_t h = tid * 4; h + 3 < H; h += stride) {
    bf16 x0 = in[x_base + h];
    bf16 x1 = in[x_base + h + 1];
    bf16 x2 = in[x_base + h + 2];
    bf16 x3 = in[x_base + h + 3];
    
    bf16 y0 = in[y_base + h];
    bf16 y1 = in[y_base + h + 1];
    bf16 y2 = in[y_base + h + 2];
    bf16 y3 = in[y_base + h + 3];
    
    float x_val0 = __bfloat162float(x0);
    float x_val1 = __bfloat162float(x1);
    float x_val2 = __bfloat162float(x2);
    float x_val3 = __bfloat162float(x3);
    
    float y_val0 = __bfloat162float(y0);
    float y_val1 = __bfloat162float(y1);
    float y_val2 = __bfloat162float(y2);
    float y_val3 = __bfloat162float(y3);
    
    float silu_x0 = silu_f(x_val0);
    float silu_x1 = silu_f(x_val1);
    float silu_x2 = silu_f(x_val2);
    float silu_x3 = silu_f(x_val3);
    
    out[out_base + h] = __float2bfloat16(silu_x0 * y_val0);
    out[out_base + h + 1] = __float2bfloat16(silu_x1 * y_val1);
    out[out_base + h + 2] = __float2bfloat16(silu_x2 * y_val2);
    out[out_base + h + 3] = __float2bfloat16(silu_x3 * y_val3);
  }
  
  // Handle remaining elements
  for (int64_t h = tid * 4; h < H; h += stride) {
    if (h + 3 >= H) {
      for (int64_t i = 0; i < 4 && h + i < H; i++) {
        int64_t idx = h + i;
        bf16 x_val_bf = in[x_base + idx];
        bf16 y_val_bf = in[y_base + idx];
        
        float x_val = __bfloat162float(x_val_bf);
        float y_val = __bfloat162float(y_val_bf);
        
        float silu_x = silu_f(x_val);
        out[out_base + idx] = __float2bfloat16(silu_x * y_val);
      }
    }
  }
}

static void fill_random(std::vector<bf16>& buf,
                        float lo=-3.f,float hi=3.f,uint32_t seed=123){
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo,hi);
  for (auto& v: buf) v = __float2bfloat16(dist(rng));
}

static void host_ref(std::vector<bf16>& out,
                     const std::vector<bf16>& in,
                     int64_t B, int64_t H){
  auto silu_h = [](double x){ return x/(1.0+std::exp(-x)); };
  for (int64_t b=0;b<B;++b){
    int64_t in_row=b*(2*H), out_row=b*H;
    for (int64_t i=0;i<H;++i){
      float x = __bfloat162float(in[in_row+i]);
      float y = __bfloat162float(in[in_row+H+i]);
      out[out_row+i] = __float2bfloat16((float)(silu_h(x)*y));
    }
  }
}

static void max_diff(const std::vector<bf16>& a,
                     const std::vector<bf16>& b,
                     double& max_abs, double& max_rel){
  max_abs=0; max_rel=0;
  for (size_t i=0;i<a.size();++i){
    double va = (double)__bfloat162float(a[i]);
    double vb = (double)__bfloat162float(b[i]);
    double ad = std::abs(va-vb);
    double rd = ad/(std::abs(vb)+1e-8);
    max_abs = std::max(max_abs, ad);
    max_rel = std::max(max_rel, rd);
  }
}

static float time_kernel_ms(std::function<void()> launch,
                            int warmup=5,int iters=100){
  hipEvent_t s,t; HIP_CHECK(hipEventCreate(&s)); HIP_CHECK(hipEventCreate(&t));
  for(int i=0;i<warmup;++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(s)); for(int i=0;i<iters;++i) launch();
  HIP_CHECK(hipEventRecord(t)); HIP_CHECK(hipEventSynchronize(t));
  float ms=0.f; HIP_CHECK(hipEventElapsedTime(&ms,s,t));
  HIP_CHECK(hipEventDestroy(s)); HIP_CHECK(hipEventDestroy(t)); return ms/iters;
}

int main(int argc, char** argv){
  int64_t B=4096, H=6400;
  for (int i=1;i<argc;++i){
    if (std::string(argv[i])=="--B" && i+1<argc) B=std::atoll(argv[++i]);
    else if (std::string(argv[i])=="--H" && i+1<argc) H=std::atoll(argv[++i]);
    else {
      printf("Usage: %s [--B <batch>] [--H <hidden>]\n", argv[0]);
      return 0;
    }
  }

  size_t in_e  = (size_t)B*(size_t)(2*H);
  size_t out_e = (size_t)B*(size_t)H;

  std::vector<bf16> h_in(in_e), h_out(out_e), h_ref(out_e);
  fill_random(h_in);

  bf16 *d_in=nullptr, *d_out=nullptr;
  HIP_CHECK(hipMalloc(&d_in,  in_e*sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_out, out_e*sizeof(bf16)));
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), in_e*sizeof(bf16), hipMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid(B);
  
  auto launch = [&](){
    hipLaunchKernelGGL(silu_mul_kernel, grid, block, 0, 0, d_out, d_in, B, H);
  };

  launch(); HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, out_e*sizeof(bf16), hipMemcpyDeviceToHost));
  host_ref(h_ref, h_in, B, H);

  double max_abs=0, max_rel=0; max_diff(h_out, h_ref, max_abs, max_rel);
  const double atol=2e-2, rtol=6e-2;
  bool ok = (max_abs <= atol) || (max_rel <= rtol);
  printf("Check: max_abs=%.4g  max_rel=%.4g  -> %s\n",
         max_abs, max_rel, ok ? "PASS":"FAIL");

  float ms = time_kernel_ms(launch, 5, 100);
  printf("Perf: %.4f ms\n", ms);

  HIP_CHECK(hipFree(d_in)); HIP_CHECK(hipFree(d_out));
}