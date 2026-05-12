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
    if (useGPU){
        cout << "Running Basic K-Means on GPU" << endl;
    }

    else{
    int numPixels = width * height;
    int dataSize = numPixels * channels * sizeof(unsigned char);

    cout << "Running Basic K-Means on CPU" << endl;
    
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
}
// }

// =====================================================================
// 2. TILED K-MEANS 
// =====================================================================
void runTiledKMeans(unsigned char* data, int width, int height, int channels, int k) {
    std::cout << "Running Tiled K-Means on GPU..." << std::endl;
    // TODO: Implement
}

// =====================================================================
// 3. FUZZY C-MEANS 
// =====================================================================
void runFuzzyCMeans(unsigned char* data, int width, int height, int channels, int k) {
    std::cout << "Running Fuzzy C-Means on GPU..." << std::endl;
    
    // Create FCMClustering instance
    // Defaults: m=2.0f, soft_clustering=true
    FCMClustering fcm(data, width, height, channels, k, 2.0f, true);
    
    // Run the algorithm
    FCMResult result = fcm.run();

    // Copy the resulting segmented image data back to the original data pointer
    int numPixels = width * height;
    for (int p = 0; p < numPixels; p++) {
        int idx = p * channels;
        data[idx + 0] = result.image_data[p * 3 + 0];
        if (channels > 1) data[idx + 1] = result.image_data[p * 3 + 1];
        if (channels > 2) data[idx + 2] = result.image_data[p * 3 + 2];
        // If there's an alpha channel, we leave it untouched.
    }
}


// =====================================================================
// 4. PARALLEL K-MEANS++ 
// =====================================================================
void runKMeansPlusPlus(unsigned char* data, int width, int height, int channels, int k) {
    std::cout << "Running Parallel K-Means++ on GPU..." << std::endl;
    // TODO: Implement
}

// =====================================================================
// 5. MINI-BATCH K-MEANS 
// =====================================================================
__global__ void miniBatchKmeansKernel(
    unsigned char* image,
    int* batch_indices,
    float* centroids,
    int* centroid_counts,
    int* labels,
    int total_pixels,
    int batch_size,
    int k, int channels)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= batch_size)
        return;

    int pixel_idx = batch_indices[tid];

    if(pixel_idx >= total_pixels)
        return;

    int rgbOffset = pixel_idx * channels;

    float r = image[rgbOffset];
    float g = image[rgbOffset + 1];
    float b = image[rgbOffset + 2];

    // Find nearest centroid

    float minDist = 1e20;
    int bestCluster = 0;

    #pragma unroll
    for(int c = 0; c < k; c++)
    {
        float cr = centroids[c * channels];
        float cg = centroids[c * channels + 1];
        float cb = centroids[c * channels + 2];

        float dist =
            (r - cr)*(r - cr) +
            (g - cg)*(g - cg) +
            (b - cb)*(b - cb);

        if(dist < minDist)
        {
            minDist = dist;
            bestCluster = c;
        }
    }

    // Store temporary batch label

    labels[pixel_idx] = bestCluster;

    // Incremental centroid update (per every assigned pixel)

    int old_count =
        atomicAdd(&centroid_counts[bestCluster], 1);

    float eta = 1.0f / (old_count + 1);

    atomicAdd(
        &centroids[bestCluster * channels],
        eta * (r - centroids[bestCluster * channels]));

    atomicAdd(
        &centroids[bestCluster * channels + 1],
        eta * (g - centroids[bestCluster * channels + 1]));

    atomicAdd(
        &centroids[bestCluster * channels + 2],
        eta * (b - centroids[bestCluster * channels + 2]));
}

// Final Full-Image Assignment Kernel

__global__ void finalAssignmentKernel(
    unsigned char* image,
    float* centroids,
    int* labels,
    int total_pixels,
    int k, int channels)
{
    int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= total_pixels)
        return;

    // Read pixel

    int rgbOffset = idx * channels;

    float r = image[rgbOffset];
    float g = image[rgbOffset + 1];
    float b = image[rgbOffset + 2];

    // Find nearest centroid

    float minDist = 1e20;
    int bestCluster = 0;

    #pragma unroll
    for(int c = 0; c < k; c++)
    {
        float cr = centroids[c * channels];
        float cg = centroids[c * channels + 1];
        float cb = centroids[c * channels + 2];

        float dist =
            (r - cr)*(r - cr) +
            (g - cg)*(g - cg) +
            (b - cb)*(b - cb);

        if(dist < minDist)
        {
            minDist = dist;
            bestCluster = c;
        }
    }

    // Store final label

    labels[idx] = bestCluster;
}

// Random mini-batch Sampling

void generateMiniBatch(
    vector<int>& batch_indices,
    int total_pixels)
{
    for(int i = 0; i < BATCH_SIZE; i++)
    {
        batch_indices[i] = rand() % total_pixels;
    }
}


void runMiniBatchKMeans(unsigned char* data, int width, int height, int channels, int k) {
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
            k, channels);

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
            k, channels);

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

// =====================================================================
// 6. NAIVE LLOYD ALGORITHM (Global Memory)
// =====================================================================

__global__ void naiveLloydKernel(
    unsigned char* image,
    float* centroids,
    int* labels,
    int total_pixels,
    int k,
    int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pixels)
        return;

    int rgbOffset = idx * channels;

    float r = image[rgbOffset];
    float g = image[rgbOffset + 1];
    float b = image[rgbOffset + 2];

    // Find nearest centroid (reading from GLOBAL memory)
    float minDist = 1e20f;
    int bestCluster = 0;

    for (int c = 0; c < k; c++)
    {
        float cr = centroids[c * channels];
        float cg = centroids[c * channels + 1];
        float cb = centroids[c * channels + 2];

        float dist =
            (r - cr) * (r - cr) +
            (g - cg) * (g - cg) +
            (b - cb) * (b - cb);

        if (dist < minDist)
        {
            minDist = dist;
            bestCluster = c;
        }
    }

    labels[idx] = bestCluster;
}

void runNaiveLloyd(unsigned char* data, int width, int height, int channels, int k) {
    std::cout << "Running Naive Lloyd Algorithm on GPU..." << std::endl;
    srand(time(NULL));

    int total_pixels = width * height;

    // Initialize centroids randomly
    float* h_centroids = new float[k * channels];

    for (int c = 0; c < k; c++)
    {
        int rand_pixel = rand() % total_pixels;
        h_centroids[c * channels] = data[rand_pixel * channels];
        h_centroids[c * channels + 1] = data[rand_pixel * channels + 1];
        h_centroids[c * channels + 2] = data[rand_pixel * channels + 2];
    }

    int* h_labels = (int*)malloc(total_pixels * sizeof(int));

    unsigned char* d_image;
    float* d_centroids;
    int* d_labels;

    cudaMalloc(&d_image, total_pixels * channels * sizeof(unsigned char));
    cudaMalloc(&d_centroids, k * channels * sizeof(float));
    cudaMalloc(&d_labels, total_pixels * sizeof(int));

    cudaMemcpy(d_image, data, total_pixels * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, k * channels * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    // Lloyd Algorithm iterations
    for (int iter = 0; iter < MAX_KMEANS_ITERS; iter++)
    {
        naiveLloydKernel<<<blocks, threadsPerBlock>>>(
            d_image,
            d_centroids,
            d_labels,
            total_pixels,
            k,
            channels);

        cudaDeviceSynchronize();

        // Copy labels back for centroid recalculation
        cudaMemcpy(h_labels, d_labels, total_pixels * sizeof(int), cudaMemcpyDeviceToHost);

        // Recalculate centroids on CPU
        vector<float> sumR(k, 0.0f);
        vector<float> sumG(k, 0.0f);
        vector<float> sumB(k, 0.0f);
        vector<int> counts(k, 0);

        for (int p = 0; p < total_pixels; p++)
        {
            int cluster = h_labels[p];
            sumR[cluster] += data[p * channels];
            sumG[cluster] += data[p * channels + 1];
            sumB[cluster] += data[p * channels + 2];
            counts[cluster]++;
        }

        bool changed = false;
        for (int c = 0; c < k; c++)
        {
            if (counts[c] > 0)
            {
                float newR = sumR[c] / counts[c];
                float newG = sumG[c] / counts[c];
                float newB = sumB[c] / counts[c];

                if (abs(newR - h_centroids[c * channels]) > 0.5f ||
                    abs(newG - h_centroids[c * channels + 1]) > 0.5f ||
                    abs(newB - h_centroids[c * channels + 2]) > 0.5f)
                {
                    changed = true;
                }

                h_centroids[c * channels] = newR;
                h_centroids[c * channels + 1] = newG;
                h_centroids[c * channels + 2] = newB;
            }
        }

        cudaMemcpy(d_centroids, h_centroids, k * channels * sizeof(float), cudaMemcpyHostToDevice);

        cout << "Iteration " << iter << " completed (changed: " << changed << ")" << endl;

        if (!changed) break;
    }

    // Final assignment and image reconstruction
    naiveLloydKernel<<<blocks, threadsPerBlock>>>(
        d_image,
        d_centroids,
        d_labels,
        total_pixels,
        k,
        channels);

    cudaDeviceSynchronize();
    cudaMemcpy(h_labels, d_labels, total_pixels * sizeof(int), cudaMemcpyDeviceToHost);

    for (int p = 0; p < total_pixels; p++)
    {
        int cluster = h_labels[p];
        data[p * channels] = (unsigned char)h_centroids[cluster * channels];
        data[p * channels + 1] = (unsigned char)h_centroids[cluster * channels + 1];
        data[p * channels + 2] = (unsigned char)h_centroids[cluster * channels + 2];
    }

    free(h_labels);
    delete[] h_centroids;

    cudaFree(d_image);
    cudaFree(d_centroids);
    cudaFree(d_labels);

    cout << "Naive Lloyd Algorithm completed!" << endl;
}

// =====================================================================
// 7. OPTIMIZED LLOYD ALGORITHM (Fused Kernels + Minimal GPU-CPU Transfers)
// =====================================================================
// PURPOSE: GPU-persistent K-means with three fused kernels to minimize
//          CPU-GPU data transfers and maximize GPU utilization
//
// OPTIMIZATION STRATEGY:
//   - Kernel 1: Fused assignment + partial reduction using shared memory
//   - Kernel 2: Global reduction combining partial results from all blocks
//   - Kernel 3: GPU-side centroid update (avoids CPU computation)
// =====================================================================

__global__ void kmeans_fused_kernel(
    unsigned char* image,
    float* centroids,
    int* labels,
    float* partial_sums_r,
    float* partial_sums_g,
    float* partial_sums_b,
    int* partial_counts,
    int total_pixels,
    int k,
    int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Allocate shared memory for this block's accumulation
    extern __shared__ char shared_mem[];
    float* shared_sums_r = (float*)shared_mem;
    float* shared_sums_g = (float*)(shared_mem + k * sizeof(float) * blockDim.x);
    float* shared_sums_b = (float*)(shared_mem + 2 * k * sizeof(float) * blockDim.x);
    int* shared_counts = (int*)(shared_mem + 3 * k * sizeof(float) * blockDim.x);

    // Initialize shared memory (all threads participate)
    for (int i = tid; i < k; i += blockDim.x)
    {
        shared_sums_r[i] = 0.0f;
        shared_sums_g[i] = 0.0f;
        shared_sums_b[i] = 0.0f;
        shared_counts[i] = 0;
    }
    __syncthreads();

    // PHASE 1: Assignment + Local Accumulation
    if (idx < total_pixels)
    {
        int rgbOffset = idx * channels;
        float r = image[rgbOffset];
        float g = image[rgbOffset + 1];
        float b = image[rgbOffset + 2];

        // Find closest centroid
        float minDist = 1e20f;
        int bestCluster = 0;

        for (int c = 0; c < k; c++)
        {
            float cr = centroids[c * channels];
            float cg = centroids[c * channels + 1];
            float cb = centroids[c * channels + 2];

            float dist =
                (r - cr) * (r - cr) +
                (g - cg) * (g - cg) +
                (b - cb) * (b - cb);

            if (dist < minDist)
            {
                minDist = dist;
                bestCluster = c;
            }
        }

        // Store assignment
        labels[idx] = bestCluster;

        // Atomically accumulate to shared memory
        atomicAdd(&shared_sums_r[bestCluster], r);
        atomicAdd(&shared_sums_g[bestCluster], g);
        atomicAdd(&shared_sums_b[bestCluster], b);
        atomicAdd(&shared_counts[bestCluster], 1);
    }
    __syncthreads();

    // PHASE 2: Write block-local results to global memory (partial results)
    for (int i = tid; i < k; i += blockDim.x)
    {
        partial_sums_r[blockIdx.x * k + i] = shared_sums_r[i];
        partial_sums_g[blockIdx.x * k + i] = shared_sums_g[i];
        partial_sums_b[blockIdx.x * k + i] = shared_sums_b[i];
        partial_counts[blockIdx.x * k + i] = shared_counts[i];
    }
}

__global__ void kmeans_reduce_kernel(
    float* partial_sums_r,
    float* partial_sums_g,
    float* partial_sums_b,
    int* partial_counts,
    float* final_sums_r,
    float* final_sums_g,
    float* final_sums_b,
    int* final_counts,
    int k,
    int num_blocks)
{
    int i = blockIdx.x;
    if (i >= k) return;

    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;
    int count_val = 0;

    // Sum across all blocks for this cluster
    for (int b = 0; b < num_blocks; b++)
    {
        sum_r += partial_sums_r[b * k + i];
        sum_g += partial_sums_g[b * k + i];
        sum_b += partial_sums_b[b * k + i];
        count_val += partial_counts[b * k + i];
    }

    // Store final aggregated results
    final_sums_r[i] = sum_r;
    final_sums_g[i] = sum_g;
    final_sums_b[i] = sum_b;
    final_counts[i] = count_val;
}

__global__ void kmeans_update_centroids_kernel(
    float* final_sums_r,
    float* final_sums_g,
    float* final_sums_b,
    int* final_counts,
    float* centroids,
    int k,
    int channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k && final_counts[i] > 0)
    {
        // Update centroid as mean of all assigned pixels
        centroids[i * channels] = final_sums_r[i] / (float)final_counts[i];
        centroids[i * channels + 1] = final_sums_g[i] / (float)final_counts[i];
        centroids[i * channels + 2] = final_sums_b[i] / (float)final_counts[i];
    }
}

void runSharedLloyd(unsigned char* data, int width, int height, int channels, int k)
{
    std::cout << "Running Optimized Lloyd Algorithm (Fused Kernels + GPU-Persistent) on GPU..." << std::endl;
    srand(time(NULL));

    int total_pixels = width * height;

    // ===== GPU MEMORY ALLOCATION =====
    unsigned char* d_image;
    float* d_centroids;
    int* d_labels;

    cudaMalloc(&d_image, total_pixels * channels * sizeof(unsigned char));
    cudaMalloc(&d_centroids, k * channels * sizeof(float));
    cudaMalloc(&d_labels, total_pixels * sizeof(int));

    // Copy image to GPU once (stays there for all iterations)
    cudaMemcpy(d_image, data, total_pixels * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Initialize centroids randomly on host
    float* h_centroids = new float[k * channels];
    for (int c = 0; c < k; c++)
    {
        int rand_pixel = rand() % total_pixels;
        h_centroids[c * channels] = data[rand_pixel * channels];
        h_centroids[c * channels + 1] = data[rand_pixel * channels + 1];
        h_centroids[c * channels + 2] = data[rand_pixel * channels + 2];
    }
    cudaMemcpy(d_centroids, h_centroids, k * channels * sizeof(float), cudaMemcpyHostToDevice);

    // ===== INTERMEDIATE BUFFERS FOR REDUCTION =====
    int threadsPerBlock = 256;
    int blocks = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    float* d_partial_sums_r;
    float* d_partial_sums_g;
    float* d_partial_sums_b;
    int* d_partial_counts;

    cudaMalloc(&d_partial_sums_r, blocks * k * sizeof(float));
    cudaMalloc(&d_partial_sums_g, blocks * k * sizeof(float));
    cudaMalloc(&d_partial_sums_b, blocks * k * sizeof(float));
    cudaMalloc(&d_partial_counts, blocks * k * sizeof(int));

    // Final aggregated buffers
    float* d_final_sums_r;
    float* d_final_sums_g;
    float* d_final_sums_b;
    int* d_final_counts;

    cudaMalloc(&d_final_sums_r, k * sizeof(float));
    cudaMalloc(&d_final_sums_g, k * sizeof(float));
    cudaMalloc(&d_final_sums_b, k * sizeof(float));
    cudaMalloc(&d_final_counts, k * sizeof(int));

    // Shared memory size for Kernel 1
    size_t shared_mem_size = k * 3 * sizeof(float) * threadsPerBlock + k * sizeof(int);

    // ===== LLOYD ITERATIONS (GPU-PERSISTENT) =====
    int* h_labels = (int*)malloc(total_pixels * sizeof(int));

    for (int iter = 0; iter < MAX_KMEANS_ITERS; iter++)
    {
        // KERNEL 1: Fused Assignment + Partial Reduction
        kmeans_fused_kernel<<<blocks, threadsPerBlock, shared_mem_size>>>(
            d_image,
            d_centroids,
            d_labels,
            d_partial_sums_r,
            d_partial_sums_g,
            d_partial_sums_b,
            d_partial_counts,
            total_pixels,
            k,
            channels);
        cudaDeviceSynchronize();

        // KERNEL 2: Global Reduction
        kmeans_reduce_kernel<<<k, 1>>>(
            d_partial_sums_r,
            d_partial_sums_g,
            d_partial_sums_b,
            d_partial_counts,
            d_final_sums_r,
            d_final_sums_g,
            d_final_sums_b,
            d_final_counts,
            k,
            blocks);
        cudaDeviceSynchronize();

        // KERNEL 3: Update Centroids on GPU
        kmeans_update_centroids_kernel<<<(k + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
            d_final_sums_r,
            d_final_sums_g,
            d_final_sums_b,
            d_final_counts,
            d_centroids,
            k,
            channels);
        cudaDeviceSynchronize();

        cout << "Iteration " << iter << " completed" << endl;
    }

    // ===== FINAL ASSIGNMENT =====
    // One final pass to assign all pixels to nearest centroid
    naiveLloydKernel<<<blocks, threadsPerBlock>>>(
        d_image,
        d_centroids,
        d_labels,
        total_pixels,
        k,
        channels);
    cudaDeviceSynchronize();

    // ===== COPY RESULTS BACK (ONLY ONCE) =====
    cudaMemcpy(h_labels, d_labels, total_pixels * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, k * channels * sizeof(float), cudaMemcpyDeviceToHost);

    // ===== IMAGE RECONSTRUCTION =====
    for (int p = 0; p < total_pixels; p++)
    {
        int cluster = h_labels[p];
        data[p * channels] = (unsigned char)h_centroids[cluster * channels];
        data[p * channels + 1] = (unsigned char)h_centroids[cluster * channels + 1];
        data[p * channels + 2] = (unsigned char)h_centroids[cluster * channels + 2];
    }

    // ===== CLEANUP =====
    free(h_labels);
    delete[] h_centroids;

    cudaFree(d_image);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_partial_sums_r);
    cudaFree(d_partial_sums_g);
    cudaFree(d_partial_sums_b);
    cudaFree(d_partial_counts);
    cudaFree(d_final_sums_r);
    cudaFree(d_final_sums_g);
    cudaFree(d_final_sums_b);
    cudaFree(d_final_counts);

    cout << "Optimized Lloyd Algorithm completed!" << endl;
}