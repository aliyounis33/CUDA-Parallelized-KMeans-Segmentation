#ifndef KERNEL_H
#define KERNEL_H
#define BATCH_SIZE 1024
#define MAX_ITERS 50

// 1. Basic K-Means
void runBasicKMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU);

// 2. Tiled K-Means
void runTiledKMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU);

// 3. Fuzzy C-Means
void runFuzzyCMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU);

// 4. Parallel K-Means++
void runKMeansPlusPlus(unsigned char* data, int width, int height, int channels, int k, bool useGPU);

// 5. Mini-Batch K-Means
void runMiniBatchKMeans(unsigned char* data, int width, int height, int channels, int k, bool useGPU);

#endif