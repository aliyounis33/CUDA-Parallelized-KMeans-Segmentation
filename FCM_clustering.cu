// ============================================================================
// FCM_clustering.cu — Full Fuzzy C-Means clustering implementation
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// STB image loading / writing (implementation in this translation unit)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "FCM_clustering.cuh"

// ============================================================================
// Helper macro for CUDA error checking
// ============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CUDA Kernels (must be defined before class methods that call them)
// ============================================================================

// Kernel 1 — Accumulate weighted pixel values for center recalculation
__global__ void fcm_accumulate_centers_kernel(
    const float* pts_r, const float* pts_g, const float* pts_b,
    const float* u,
    float* num_r, float* num_g, float* num_b, float* den,
    int n, int c, float m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int j = 0; j < c; j++) {
            float u_ij_m = powf(u[i * c + j], m);
            atomicAdd(&num_r[j], u_ij_m * pts_r[i]);
            atomicAdd(&num_g[j], u_ij_m * pts_g[i]);
            atomicAdd(&num_b[j], u_ij_m * pts_b[i]);
            atomicAdd(&den[j],   u_ij_m);
        }
    }
}

// Kernel 2 — Divide accumulated sums to get new centers
__global__ void fcm_divide_centers_kernel(
    float* centers_r, float* centers_g, float* centers_b,
    const float* num_r, const float* num_g, const float* num_b,
    const float* den, int c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < c) {
        centers_r[j] = num_r[j] / den[j];
        centers_g[j] = num_g[j] / den[j];
        centers_b[j] = num_b[j] / den[j];
    }
}

// Kernel 3 — Update membership matrix
__global__ void fcm_update_membership_kernel(
    const float* pts_r, const float* pts_g, const float* pts_b,
    const float* centers_r, const float* centers_g, const float* centers_b,
    float* u_new, int n, int c, float m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float exponent = 2.0f / (m - 1.0f);
    if (i < n) {
        for (int j = 0; j < c; j++) {
            float dr = pts_r[i] - centers_r[j];
            float dg = pts_g[i] - centers_g[j];
            float db = pts_b[i] - centers_b[j];
            float dist_ij = sqrtf(dr * dr + dg * dg + db * db) + 1e-6f;

            float sum = 0.0f;
            for (int k = 0; k < c; k++) {
                float drk = pts_r[i] - centers_r[k];
                float dgk = pts_g[i] - centers_g[k];
                float dbk = pts_b[i] - centers_b[k];
                float dist_ik = sqrtf(drk * drk + dgk * dgk + dbk * dbk) + 1e-6f;
                sum += powf(dist_ij / dist_ik, exponent);
            }
            u_new[i * c + j] = 1.0f / sum;
        }
    }
}

// ============================================================================
// FCMClustering — Constructor
// ============================================================================
FCMClustering::FCMClustering(unsigned char* image_data,
                             int   width,
                             int   height,
                             int   channels,
                             int   num_clusters,
                             float m,
                             bool  soft_clustering,
                             bool  use_device)
    : m_image_data(image_data),
      m_width(width), m_height(height), m_channels(channels),
      m_num_clusters(num_clusters),
      m_fuzziness(m),
      m_soft_clustering(soft_clustering),
      m_use_device(use_device),
      m_N(width * height),
      h_pts_r(nullptr), h_pts_g(nullptr), h_pts_b(nullptr),
      h_u(nullptr),
      h_centers_r(nullptr), h_centers_g(nullptr), h_centers_b(nullptr),
      m_output_image(nullptr),
      m_elapsed_time(0.0),
      m_has_run(false)
{
    // --- Validate fuzzification parameter ---
    if (m_fuzziness < 1.5f || m_fuzziness > 2.5f) {
        fprintf(stderr, "[FCMClustering] Warning: m = %.2f is out of recommended range [1.5, 2.5]. Clamping.\n", m_fuzziness);
        if (m_fuzziness < 1.5f) m_fuzziness = 1.5f;
        if (m_fuzziness > 2.5f) m_fuzziness = 2.5f;
    }

    // --- Validate number of clusters ---
    if (m_num_clusters < 2) {
        fprintf(stderr, "[FCMClustering] Warning: num_clusters must be >= 2. Setting to 2.\n");
        m_num_clusters = 2;
    }
}

// ============================================================================
// FCMClustering — Destructor
// ============================================================================
FCMClustering::~FCMClustering() {
    free(h_pts_r);   free(h_pts_g);   free(h_pts_b);
    free(h_u);
    free(h_centers_r); free(h_centers_g); free(h_centers_b);
    free(m_output_image);
}

// ============================================================================
// run() — Main entry point
// ============================================================================
FCMResult FCMClustering::run() {
    // 1. Load and prepare the image
    setup_data();
    init_memberships();

    // 2. Run FCM on the selected device and measure time
    clock_t t_start = clock();

    if (m_use_device) {
        run_on_gpu();
    } else {
        run_on_cpu();
    }

    clock_t t_end = clock();
    m_elapsed_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;

    // 3. Build the segmented output image (hard or soft)
    build_output_image();

    m_has_run = true;

    // 4. Package results
    FCMResult result;
    result.image_data   = m_output_image;
    result.width        = m_width;
    result.height       = m_height;
    result.type         = m_soft_clustering ? "soft" : "hard";
    result.process_on   = m_use_device      ? "GPU"  : "CPU";
    result.time_seconds = m_elapsed_time;

    // Print summary
    printf("====================================\n");
    printf("  FCM Clustering Complete\n");
    printf("------------------------------------\n");
    printf("  Image Size: %dx%d (%d channels)\n", m_width, m_height, m_channels);
    printf("  Clusters  : %d\n", m_num_clusters);
    printf("  m         : %.2f\n", m_fuzziness);
    printf("  Type      : %s\n", result.type);
    printf("  Device    : %s\n", result.process_on);
    printf("  Time      : %.4f seconds\n", result.time_seconds);
    printf("====================================\n");

    return result;
}

// ============================================================================
// save() — Write output image to disk
// ============================================================================
void FCMClustering::save(const char* output_path) {
    if (!m_has_run || !m_output_image) {
        fprintf(stderr, "[FCMClustering] Error: run() must be called before save().\n");
        return;
    }
    stbi_write_png(output_path, m_width, m_height, 3, m_output_image, m_width * 3);
    printf("Saved segmented image to: %s\n", output_path);
}

// ============================================================================
// setup_data() — Prepare float arrays from provided image data
// ============================================================================
void FCMClustering::setup_data() {
    size_t pts_bytes = m_N * sizeof(float);
    size_t u_bytes   = m_N * m_num_clusters * sizeof(float);
    size_t c_bytes   = m_num_clusters * sizeof(float);

    h_pts_r    = (float*)malloc(pts_bytes);
    h_pts_g    = (float*)malloc(pts_bytes);
    h_pts_b    = (float*)malloc(pts_bytes);
    h_u        = (float*)malloc(u_bytes);
    h_centers_r = (float*)malloc(c_bytes);
    h_centers_g = (float*)malloc(c_bytes);
    h_centers_b = (float*)malloc(c_bytes);

    // Extract RGB channels as floats
    for (int i = 0; i < m_N; i++) {
        h_pts_r[i] = (float)m_image_data[i * m_channels + 0];
        if (m_channels > 1) {
            h_pts_g[i] = (float)m_image_data[i * m_channels + 1];
        } else {
            h_pts_g[i] = h_pts_r[i];
        }
        if (m_channels > 2) {
            h_pts_b[i] = (float)m_image_data[i * m_channels + 2];
        } else {
            h_pts_b[i] = h_pts_r[i];
        }
    }

    printf("Setup data: (%dx%d, %d pixels)\n", m_width, m_height, m_N);
}

// ============================================================================
// init_memberships() — Random normalized initialization of U matrix
// ============================================================================
void FCMClustering::init_memberships() {
    srand(42);
    int C = m_num_clusters;

    for (int i = 0; i < m_N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float val = (float)rand() / RAND_MAX;
            h_u[i * C + j] = val;
            sum += val;
        }
        for (int j = 0; j < C; j++) {
            h_u[i * C + j] /= sum;
        }
    }
}

// ============================================================================
// color_distance() — Euclidean distance in RGB space (with epsilon)
// ============================================================================
float FCMClustering::color_distance(float r1, float g1, float b1,
                                     float r2, float g2, float b2) {
    float dr = r1 - r2;
    float dg = g1 - g2;
    float db = b1 - b2;
    return sqrtf(dr * dr + dg * dg + db * db) + 1e-6f;
}

// ============================================================================
// run_on_cpu() — Full FCM iteration on the host
// ============================================================================
void FCMClustering::run_on_cpu() {
    int C = m_num_clusters;
    float m = m_fuzziness;
    size_t u_bytes = m_N * C * sizeof(float);
    float* u_new = (float*)malloc(u_bytes);

    int iter = 0;
    float max_diff;

    do {
        max_diff = 0.0f;

        // --- Step 1: Recalculate cluster centers ---
        for (int j = 0; j < C; j++) {
            float num_r = 0.0f, num_g = 0.0f, num_b = 0.0f, den = 0.0f;
            for (int i = 0; i < m_N; i++) {
                float u_ij_m = powf(h_u[i * C + j], m);
                num_r += u_ij_m * h_pts_r[i];
                num_g += u_ij_m * h_pts_g[i];
                num_b += u_ij_m * h_pts_b[i];
                den   += u_ij_m;
            }
            h_centers_r[j] = num_r / den;
            h_centers_g[j] = num_g / den;
            h_centers_b[j] = num_b / den;
        }

        // --- Step 2: Update membership matrix ---
        float exponent = 2.0f / (m - 1.0f);
        for (int i = 0; i < m_N; i++) {
            for (int j = 0; j < C; j++) {
                float dist_ij = color_distance(h_pts_r[i], h_pts_g[i], h_pts_b[i],
                                               h_centers_r[j], h_centers_g[j], h_centers_b[j]);
                float sum = 0.0f;
                for (int k = 0; k < C; k++) {
                    float dist_ik = color_distance(h_pts_r[i], h_pts_g[i], h_pts_b[i],
                                                   h_centers_r[k], h_centers_g[k], h_centers_b[k]);
                    sum += powf(dist_ij / dist_ik, exponent);
                }
                u_new[i * C + j] = 1.0f / sum;

                float diff = fabsf(u_new[i * C + j] - h_u[i * C + j]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        // Copy new memberships
        for (int i = 0; i < m_N * C; i++) {
            h_u[i] = u_new[i];
        }

        iter++;
    } while (max_diff > EPSILON && iter < MAX_ITER);

    printf("CPU finished in %d iterations.\n", iter);
    free(u_new);
}

// ============================================================================
// run_on_gpu() — Full FCM iteration on the device
// ============================================================================
void FCMClustering::run_on_gpu() {
    int C = m_num_clusters;
    float m = m_fuzziness;

    float* d_pts_r, *d_pts_g, *d_pts_b;
    float* d_u, *d_u_new;
    float* d_centers_r, *d_centers_g, *d_centers_b;
    float* d_num_r, *d_num_g, *d_num_b, *d_den;

    size_t pts_bytes = m_N * sizeof(float);
    size_t u_bytes   = m_N * C * sizeof(float);
    size_t c_bytes   = C * sizeof(float);

    // 1. Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_pts_r, pts_bytes));
    CUDA_CHECK(cudaMalloc(&d_pts_g, pts_bytes));
    CUDA_CHECK(cudaMalloc(&d_pts_b, pts_bytes));
    CUDA_CHECK(cudaMalloc(&d_u,     u_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_new, u_bytes));
    CUDA_CHECK(cudaMalloc(&d_centers_r, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_centers_g, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_centers_b, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_num_r, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_num_g, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_num_b, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_den,   c_bytes));

    // 2. Copy data to device
    CUDA_CHECK(cudaMemcpy(d_pts_r, h_pts_r, pts_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pts_g, h_pts_g, pts_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pts_b, h_pts_b, pts_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u,     h_u,     u_bytes,   cudaMemcpyHostToDevice));

    int threads  = 256;
    int blocks_N = (m_N + threads - 1) / threads;
    int blocks_C = (C   + threads - 1) / threads;

    int iter = 0;
    float max_diff;
    float* h_u_new = (float*)malloc(u_bytes);

    do {
        // Zero accumulators
        CUDA_CHECK(cudaMemset(d_num_r, 0, c_bytes));
        CUDA_CHECK(cudaMemset(d_num_g, 0, c_bytes));
        CUDA_CHECK(cudaMemset(d_num_b, 0, c_bytes));
        CUDA_CHECK(cudaMemset(d_den,   0, c_bytes));

        // Step 1: Calculate centers
        fcm_accumulate_centers_kernel<<<blocks_N, threads>>>(
            d_pts_r, d_pts_g, d_pts_b, d_u,
            d_num_r, d_num_g, d_num_b, d_den,
            m_N, C, m);
        cudaDeviceSynchronize();

        fcm_divide_centers_kernel<<<blocks_C, threads>>>(
            d_centers_r, d_centers_g, d_centers_b,
            d_num_r, d_num_g, d_num_b, d_den, C);
        cudaDeviceSynchronize();

        // Step 2: Update memberships
        fcm_update_membership_kernel<<<blocks_N, threads>>>(
            d_pts_r, d_pts_g, d_pts_b,
            d_centers_r, d_centers_g, d_centers_b,
            d_u_new, m_N, C, m);
        cudaDeviceSynchronize();

        // Step 3: Check convergence
        CUDA_CHECK(cudaMemcpy(h_u_new, d_u_new, u_bytes, cudaMemcpyDeviceToHost));
        max_diff = 0.0f;
        for (int i = 0; i < m_N * C; i++) {
            float diff = fabsf(h_u_new[i] - h_u[i]);
            if (diff > max_diff) max_diff = diff;
            h_u[i] = h_u_new[i];
        }

        // Swap device pointers
        float* temp = d_u;
        d_u     = d_u_new;
        d_u_new = temp;

        iter++;
    } while (max_diff > EPSILON && iter < MAX_ITER);

    // Copy final centers back
    CUDA_CHECK(cudaMemcpy(h_centers_r, d_centers_r, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_centers_g, d_centers_g, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_centers_b, d_centers_b, c_bytes, cudaMemcpyDeviceToHost));

    printf("GPU finished in %d iterations.\n", iter);

    // Cleanup device memory
    free(h_u_new);
    cudaFree(d_pts_r); cudaFree(d_pts_g); cudaFree(d_pts_b);
    cudaFree(d_u);     cudaFree(d_u_new);
    cudaFree(d_centers_r); cudaFree(d_centers_g); cudaFree(d_centers_b);
    cudaFree(d_num_r); cudaFree(d_num_g); cudaFree(d_num_b); cudaFree(d_den);
}

// ============================================================================
// build_output_image() — Generate segmented pixels (hard or soft)
// ============================================================================
void FCMClustering::build_output_image() {
    int C = m_num_clusters;
    m_output_image = (unsigned char*)malloc(m_N * 3);

    if (m_soft_clustering) {
        // Soft: weighted average of all cluster centers by membership
        for (int i = 0; i < m_N; i++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (int j = 0; j < C; j++) {
                float u_ij = h_u[i * C + j];
                r += u_ij * h_centers_r[j];
                g += u_ij * h_centers_g[j];
                b += u_ij * h_centers_b[j];
            }
            m_output_image[i * 3 + 0] = (unsigned char)fminf(fmaxf(r, 0.0f), 255.0f);
            m_output_image[i * 3 + 1] = (unsigned char)fminf(fmaxf(g, 0.0f), 255.0f);
            m_output_image[i * 3 + 2] = (unsigned char)fminf(fmaxf(b, 0.0f), 255.0f);
        }
    } else {
        // Hard: assign pixel to the single highest-membership cluster
        for (int i = 0; i < m_N; i++) {
            int   best = 0;
            float max_u = -1.0f;
            for (int j = 0; j < C; j++) {
                if (h_u[i * C + j] > max_u) {
                    max_u = h_u[i * C + j];
                    best  = j;
                }
            }
            m_output_image[i * 3 + 0] = (unsigned char)fminf(fmaxf(h_centers_r[best], 0.0f), 255.0f);
            m_output_image[i * 3 + 1] = (unsigned char)fminf(fmaxf(h_centers_g[best], 0.0f), 255.0f);
            m_output_image[i * 3 + 2] = (unsigned char)fminf(fmaxf(h_centers_b[best], 0.0f), 255.0f);
        }
    }
}
