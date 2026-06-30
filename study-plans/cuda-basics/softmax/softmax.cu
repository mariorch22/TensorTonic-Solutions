#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    float max = -INFINITY;
    for (int i = 0; i < N; i++) {
        max = fmax(max, input[i]);
    };

    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += expf(input[i] - max);
    };
    
    output[idx] = expf(input[idx] - max) / sum;
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}