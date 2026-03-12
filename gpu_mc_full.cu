#include <iostream>
#include <cmath>
#include <curand_kernel.h>
#include <chrono>

__global__ void monteCarloKernel(
    float *results,
    float S0,
    float K,
    float r,
    float sigma,
    float T,
    int N,
    unsigned long long seed,
    int simsPerThread
) {
    extern __shared__ float sharedSums[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;
    if(idx >= N) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float localSum = 0.0f;
    for (int i = 0; i < simsPerThread; i++) {
        float Z = curand_normal(&state);

        float ST = S0 * expf((r - 0.5f*sigma*sigma)*T + sigma*sqrtf(T)*Z);

        localSum += fmaxf(ST - K, 0.0f);
    }

    sharedSums[localIdx] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s>>=1) {
        if (localIdx < s) {
            sharedSums[localIdx] += sharedSums[localIdx + s];
        }
        __syncthreads();
    }

    if (localIdx == 0) {
        results[blockIdx.x] = sharedSums[localIdx];
    }
}

int main() {
    int N = 1 << 20; 
    float S0 = 100.0f;
    float K  = 100.0f;
    float r  = 0.05f;
    float sigma = 0.2f;
    float T  = 1.0f;

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int simsPerThread = N / (threadsPerBlock * numBlocks);

    float *h_results = new float[numBlocks];
    float* d_results;
    cudaMalloc (&d_results, numBlocks * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();

    monteCarloKernel <<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>> (d_results, S0, K, r, sigma, T, N, 1234ULL, simsPerThread);
    cudaMemcpy(h_results, d_results, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for(int i = 0; i < numBlocks; i++)
        sum += h_results[i];

    double price = exp(-r*T) * (sum / N);

    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output
    std::cout << "European Call Option Price: " << price << std::endl;
    std::cout << "GPU Monte Carlo took: " << elapsed.count() << " seconds" << std::endl;

    cudaFree(d_results);
    delete[] h_results;

    return 0;

}