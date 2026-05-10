#include "kernel.h"
#include <cuda_runtime.h>
#include <iostream>

// =====================================================================
// 1. BASIC K-MEANS
// =====================================================================
__global__ void basicDummyKernel(unsigned char* data, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        data[idx * 3] = 255; // Dummy GPU operation: Paint it red
    }
}

void runBasicKMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU) {
    int numPixels = width * height;
    int dataSize = numPixels * channels * sizeof(unsigned char);

    if (useGPU) {
        std::cout << "Running Basic K-Means on GPU..." << std::endl;
        unsigned char* d_data;
        cudaMalloc((void**)&d_data, dataSize);
        cudaMemcpy(d_data, data, dataSize, cudaMemcpyHostToDevice);
        
        int threads = 256;
        int blocks = (numPixels + threads - 1) / threads;
        basicDummyKernel<<<blocks, threads>>>(d_data, numPixels);
        
        cudaDeviceSynchronize();
        cudaMemcpy(data, d_data, dataSize, cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    } else {
        std::cout << "Running Basic K-Means on CPU..." << std::endl;
        for (int i = 0; i < numPixels; ++i) {
            data[i * channels + 1] = 255; // Dummy CPU operation: Paint it green
        }
    }
}

// =====================================================================
// 2. TILED K-MEANS 
// =====================================================================
void runTiledKMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU) {
    if (useGPU) std::cout << "Running Tiled K-Means on GPU..." << std::endl;
    else        std::cout << "Running Tiled K-Means on CPU..." << std::endl;
    // TODO: Implement
}

// =====================================================================
// 3. FUZZY C-MEANS 
// =====================================================================
void runFuzzyCMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU) {
    if (useGPU) std::cout << "Running Fuzzy C-Means on GPU..." << std::endl;
    else        std::cout << "Running Fuzzy C-Means on CPU..." << std::endl;
    // TODO: Implement
}

// =====================================================================
// 4. PARALLEL K-MEANS++ 
// =====================================================================
void runKMeansPlusPlus(unsigned char* data, int width, int height, int channels, int k, bool useGPU) {
    if (useGPU) std::cout << "Running Parallel K-Means++ on GPU..." << std::endl;
    else        std::cout << "Running Parallel K-Means++ on CPU..." << std::endl;
    // TODO: Implement
}

// =====================================================================
// 5. MINI-BATCH K-MEANS 
// =====================================================================
void runMiniBatchKMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU) {
    if (useGPU) std::cout << "Running Mini-Batch K-Means on GPU..." << std::endl;
    else        std::cout << "Running Mini-Batch K-Means on CPU..." << std::endl;
    // TODO: Implement
}