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
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float Z = curand_normal(&state);

    float ST = S0 * expf((r - 0.5f*sigma*sigma)*T + sigma*sqrtf(T)*Z);

    results[idx] = fmaxf(ST - K, 0.0f);
}

int main() {
    int N = 1 << 20; 
    float S0 = 100.0f;
    float K  = 100.0f;
    float r  = 0.05f;
    float sigma = 0.2f;
    float T  = 1.0f;

    float* d_results;
    float *h_results = new float[N];
    cudaMalloc (&d_results, N * sizeof(float));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    monteCarloKernel <<<numBlocks, threadsPerBlock>>> (d_results, S0, K, r, sigma, T, N, 1234ULL);
    cudaMemcpy(h_results, d_results, N * sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for(int i = 0; i < N; i++)
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