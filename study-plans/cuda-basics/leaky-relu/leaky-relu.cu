#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    if (input[idx] < 0) {
        output[idx] = alpha * input[idx];
    } else {
        output[idx] = input[idx];
    };
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}