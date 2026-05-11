#ifndef KERNEL_H
#define KERNEL_H
#include <vector>

#define BATCH_SIZE 1024
#define MAX_ITERS 50
#define MAX_KMEANS_ITERS 100

struct Centroid {
    float r, g, b; // RGB components
};

// 1. Basic K-Means
void runBasicKMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU);

// 2. Tiled K-Means
void runTiledKMeans(unsigned char* data, int width, int height, int channels, int k);

// 3. Fuzzy C-Means
void runFuzzyCMeans(unsigned char* data, int width, int height, int channels, int k);

// 4. Parallel K-Means++
void runKMeansPlusPlus(unsigned char* data, int width, int height, int channels, int k);

// 5. Mini-Batch K-Means
void runMiniBatchKMeans(unsigned char* data, int width, int height, int channels, int k);

// 6. Naive Lloyd Algorithm (Global Memory)
void runNaiveLloyd(unsigned char* data, int width, int height, int channels, int k);

// 7. Shared Memory Lloyd Algorithm
void runSharedLloyd(unsigned char* data, int width, int height, int channels, int k);

#endif