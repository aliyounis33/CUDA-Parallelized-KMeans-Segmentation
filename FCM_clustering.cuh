#pragma once

#include <cuda_runtime.h>

// ============================================================================
// FCMResult — returned by FCMClustering::run()
// ============================================================================
struct FCMResult {
    unsigned char* image_data;   // Output segmented image pixels (RGB, width*height*3)
    int            width;
    int            height;
    const char*    type;         // "soft" or "hard"
    const char*    process_on;   // "GPU" or "CPU"
    double         time_seconds; // Time spent in the clustering step only
};

// ============================================================================
// FCMClustering — Fuzzy C-Means image segmentation
// ============================================================================
//
// Usage:
//   FCMClustering fcm("images/photo.png");            // all defaults
//   FCMClustering fcm("path.png", 5, 2.0f, true, true); // custom
//   FCMResult result = fcm.run();
//   fcm.save("output.png");
//
// Constructor parameters:
//   image_data          — pointer to image pixel data (required)
//   width               — image width (required)
//   height              — image height (required)
//   channels            — image channels (default 3)
//   num_clusters        — number of clusters           (default 3)
//   m                   — fuzzification coefficient    (default 2.0, range [1.5, 2.5])
//   soft_clustering     — true = soft, false = hard    (default true  → soft)
//   use_device          — true = GPU,  false = CPU     (default true  → GPU)
// ============================================================================

class FCMClustering {
public:
    FCMClustering(unsigned char* image_data,
                  int   width,
                  int   height,
                  int   channels        = 3,
                  int   num_clusters    = 3,
                  float m               = 2.0f,
                  bool  soft_clustering = true,
                  bool  use_device      = true);
    ~FCMClustering();

    // Run the algorithm and return results
    FCMResult run();

    // Convenience: save the output image after run()
    void save(const char* output_path);

private:
    // --- Configuration ---
    unsigned char* m_image_data;
    int         m_channels;
    int         m_num_clusters;
    float       m_fuzziness;
    bool        m_soft_clustering;
    bool        m_use_device;

    // --- Constants ---
    static constexpr float  EPSILON  = 0.0001f;
    static constexpr int    MAX_ITER = 100;

    // --- Image info ---
    int m_width, m_height, m_N;

    // --- Host buffers ---
    float* h_pts_r;     float* h_pts_g;     float* h_pts_b;
    float* h_u;
    float* h_centers_r; float* h_centers_g; float* h_centers_b;

    // --- Output ---
    unsigned char* m_output_image;
    double         m_elapsed_time;
    bool           m_has_run;

    // --- Internal helpers ---
    void setup_data();
    void init_memberships();
    void run_on_cpu();
    void run_on_gpu();
    void build_output_image();

    // CPU distance helper
    static float color_distance(float r1, float g1, float b1,
                                float r2, float g2, float b2);
};
