#include "kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace std;

// =====================================================================
// 1. BASIC K-MEANS
// =====================================================================

void runBasicKMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU) {
    int numPixels = width * height;
    int dataSize = numPixels * channels * sizeof(unsigned char);

    cout << "Running Basic K-Means on CPU (Lloyd Algorithm)..." << endl;
    
    // ===== HELPER FUNCTION =====
    auto getSquaredDistance = [](const unsigned char* pixel, const Centroid& c) {
        float dr = pixel[0] - c.r;
        float dg = pixel[1] - c.g;
        float db = pixel[2] - c.b;
        return dr * dr + dg * dg + db * db;
    };
    
    // ===== SEQUENTIAL LLOYD ALGORITHM =====
    vector<int> labels(numPixels, 0);
    
    // Initialize centroids by randomly selecting K pixels
    vector<Centroid> centroids(k);
    srand(time(nullptr));
    
    for (int i = 0; i < k; i++) {
        int randomIdx = rand() % numPixels;
        int pixelOffset = randomIdx * channels;
        centroids[i].r = data[pixelOffset];
        centroids[i].g = data[pixelOffset + 1];
        centroids[i].b = data[pixelOffset + 2];
    }
    
    // Accumulators for centroid recalculation
    vector<float> sumR(k, 0.0f);
    vector<float> sumG(k, 0.0f);
    vector<float> sumB(k, 0.0f);
    vector<int> counts(k, 0);
    
    bool changed = true;
    int iter = 0;
    
    cout << "Total pixels: " << numPixels << ", K: " << k << endl;
    
    // Main algorithm loop
    while (changed && iter < MAX_KMEANS_ITERS) {
        changed = false;
        
        // Reset accumulators
        fill(sumR.begin(), sumR.end(), 0.0f);
        fill(sumG.begin(), sumG.end(), 0.0f);
        fill(sumB.begin(), sumB.end(), 0.0f);
        fill(counts.begin(), counts.end(), 0);
        
        // Step 1: Assign each pixel to the nearest centroid
        for (int p = 0; p < numPixels; p++) {
            unsigned char* pixel = &data[p * channels];
            
            float minDist = 1e18f;
            int bestCluster = 0;
            
            // Find closest centroid
            for (int c = 0; c < k; c++) {
                float dist = getSquaredDistance(pixel, centroids[c]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = c;
                }
            }
            
            // Check if assignment changed
            if (labels[p] != bestCluster) {
                changed = true;
                labels[p] = bestCluster;
            }
            
            // Accumulate pixel values for centroid recalculation
            sumR[bestCluster] += pixel[0];
            sumG[bestCluster] += pixel[1];
            sumB[bestCluster] += pixel[2];
            counts[bestCluster]++;
        }
        
        // Step 2: Recalculate centroids
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                centroids[c].r = sumR[c] / counts[c];
                centroids[c].g = sumG[c] / counts[c];
                centroids[c].b = sumB[c] / counts[c];
            }
        }
        
        iter++;
        cout << "Iteration " << iter << " completed (changed: " << changed << ")" << endl;
    }
    
    cout << "K-Means converged in " << iter << " iterations." << endl;
    
    // Reconstruct the image with centroid colors
    for (int p = 0; p < numPixels; p++) {
        int clusterIdx = labels[p];
        int pixelOffset = p * channels;
        data[pixelOffset] = static_cast<unsigned char>(centroids[clusterIdx].r);
        data[pixelOffset + 1] = static_cast<unsigned char>(centroids[clusterIdx].g);
        data[pixelOffset + 2] = static_cast<unsigned char>(centroids[clusterIdx].b);
    }
}
// }

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