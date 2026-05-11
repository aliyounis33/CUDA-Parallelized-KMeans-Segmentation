#include "kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <ctime>

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
    if (useGPU) {
    std::cout << "Running Mini-Batch K-Means on GPU..." << std::endl;
    srand(time(NULL));

    int total_pixels = width * height;

    int* h_labels =
        (int*)malloc(total_pixels * sizeof(int));

    // Initialize centroids randomly

    float* h_centroids =
        new float[k * channels];

    for(int c = 0; c < k; c++)
    {
        int rand_pixel = rand() % total_pixels;

        h_centroids[c * channels] =
            data[rand_pixel * channels];

        h_centroids[c * channels + 1] =
            data[rand_pixel * channels + 1];

        h_centroids[c * channels + 2] =
            data[rand_pixel * channels + 2];
    }

    // Cluster counts

    int* h_counts = new int[k];

    for(int i = 0; i < k; i++)
        h_counts[i] = 1;

    // Mini-batch indices that will be copied (not the batch pixels)

    vector<int> batch_indices(BATCH_SIZE);

    unsigned char* d_image;
    float* d_centroids;
    int* d_counts;
    int* d_labels;
    int* d_batch_indices;

    cudaMalloc(
        &d_image,
        total_pixels * channels * sizeof(unsigned char));

    cudaMalloc(
        &d_centroids,
        k * channels * sizeof(float));

    cudaMalloc(
        &d_counts,
        k * sizeof(int));

    cudaMalloc(
        &d_labels,
        total_pixels * sizeof(int));

    cudaMalloc(
        &d_batch_indices,
        BATCH_SIZE * sizeof(int));

    cudaMemcpy(
        d_image,
        data,
        total_pixels * channels * sizeof(unsigned char),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_centroids,
        h_centroids,
        k * channels * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_counts,
        h_counts,
        k * sizeof(int),
        cudaMemcpyHostToDevice);

    // CUDA configuration

    int threadsPerBlock = 256;

    int blocks =
        (BATCH_SIZE + threadsPerBlock - 1)
        / threadsPerBlock;

    // Mini-Batch K-Means Iterations

    for(int iter = 0; iter < MAX_ITERS; iter++)
    {
        generateMiniBatch(batch_indices, total_pixels);

        cudaMemcpy(
            d_batch_indices,
            batch_indices.data(),
            BATCH_SIZE * sizeof(int),
            cudaMemcpyHostToDevice);

        miniBatchKmeansKernel<<<blocks, threadsPerBlock>>>(
            d_image,
            d_batch_indices,
            d_centroids,
            d_counts,
            d_labels,
            total_pixels,
            BATCH_SIZE,
            k);

        cudaDeviceSynchronize();

        cout << "Iteration "
             << iter
             << " completed"
             << endl;
    }

    // Final Full-Image Assignment

    int finalThreads = 256;

    int finalBlocks =
        (total_pixels + finalThreads - 1)
        / finalThreads;

    finalAssignmentKernel<<<
        finalBlocks,
        finalThreads>>>(
            d_image,
            d_centroids,
            d_labels,
            total_pixels,
            k);

    cudaDeviceSynchronize();

    // Copy final labels and centroids back to host

    cudaMemcpy(
        h_labels,
        d_labels,
        total_pixels * sizeof(int),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        h_centroids,
        d_centroids,
        k * channels * sizeof(float),
        cudaMemcpyDeviceToHost);

    // Create the segmented image and save it

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            int idx = y * width + x;

            int cluster = h_labels[idx];

            data[idx * channels] =
                (unsigned char)h_centroids[cluster * channels];

            data[idx * channels + 1] =
                (unsigned char)h_centroids[cluster * channels + 1];

            data[idx * channels + 2] =
                (unsigned char)h_centroids[cluster * channels + 2];
        }
    }

    cout << "Segmented image saved successfully!"
         << endl;

    // Free the allocated GPU memory

    free(h_labels);

    delete[] h_centroids;
    delete[] h_counts;

    cudaFree(d_image);
    cudaFree(d_centroids);
    cudaFree(d_counts);
    cudaFree(d_labels);
    cudaFree(d_batch_indices);
    }

    else {       
    std::cout << "Running Mini-Batch K-Means on CPU..." << std::endl;
    // TODO: Implement
    }
}