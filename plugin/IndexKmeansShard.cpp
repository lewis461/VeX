/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "IndexKmeansShard.h"

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>
#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>
#include <memory>
#include <set>
#include <cstring>  // for memset
#include <chrono>   // for timing
#include <atomic>   // for progress tracking
#include <omp.h>    // for parallel search

// ============================================================================
// KH: Import mode configuration from IndexShardBase.cpp
// These must match the definitions in IndexShardBase.cpp
// ============================================================================

// Uncomment to enable boundary vector overlap
#define PARTITION_NESTED

// ============================================================================
// PARTITION_NESTED Parameters (distance-based boundary detection)
// ============================================================================
// A vector is considered a "boundary vector" if:
//   distance_to_2nd_nearest / distance_to_1st_nearest < BOUNDARY_RATIO_THRESHOLD
// This means it's nearly equidistant to two centroids.
//
// BOUNDARY_RATIO_THRESHOLD: Lower value = more selective (fewer boundary vectors)
//   - 1.05 means 2nd nearest is at most 5% farther than nearest
//   - 1.10 means 2nd nearest is at most 10% farther than nearest
//   - 1.20 means 2nd nearest is at most 20% farther than nearest
// ============================================================================
#ifndef BOUNDARY_RATIO_THRESHOLD
constexpr float BOUNDARY_RATIO_THRESHOLD = 1.01f;
#endif

// ============================================================================

namespace faiss {

IndexKmeansShard::IndexKmeansShard(int d, MetricType metric)
        : IndexShardBase(d, metric), niter(25) {
    printf("IndexKmeansShard constructor: d=%d, metric=%d\n", d, (int)metric);
}

void IndexKmeansShard::train(idx_t n, const float* x) {
    fprintf(stderr, "IndexKmeansShard::train called with n=%zd, d=%d, num_shards=%d, verbose=%d\n",
           n, d, num_shards, verbose);
    fflush(stderr);

    FAISS_THROW_IF_NOT_MSG(n > 0, "Cannot train on empty dataset");
    FAISS_THROW_IF_NOT_MSG(x, "Training data is null");
    FAISS_THROW_IF_NOT_MSG(num_shards > 0, "num_shards must be set before training");

    // KH: Print build mode configuration
    printf("============================================================\n");
    printf("IndexKmeansShard BUILD MODE CONFIGURATION:\n");
#ifdef SAMPLING_KMEANS
    printf("  [SAMPLING]   SAMPLING_KMEANS: ENABLED (sub-clustering)\n");
#else
    printf("  [SAMPLING]   SAMPLING_KMEANS: DISABLED (random sampling)\n");
#endif
#ifdef PARTITION_NESTED
    printf("  [PARTITION]  PARTITION_NESTED: ENABLED (boundary overlap)\n");
    printf("               - Boundary ratio threshold: %.2f\n", BOUNDARY_RATIO_THRESHOLD);
#else
    printf("  [PARTITION]  PARTITION_NESTED: DISABLED (no overlap)\n");
#endif
    printf("============================================================\n");

    if (verbose) {
        printf("IndexKmeansShard: training on %zd vectors with %d shards\n",
               n, num_shards);
    }

    // Perform k-means partitioning
    std::vector<int> assignments = kmeans_partition(n, x);

    // KH: Compute shard centroids or sample random vectors (based on num_samples_per_shard)
    // Note: kmeans_partition already sets shard_centroids from k-means,
    // but we allow overriding with random sampling if num_samples_per_shard > 0
    if (num_samples_per_shard > 0) {
        // KH: Override k-means centroids with random sampling
        compute_shard_centroids(x, n, assignments);
    } else {
        // KH: Centroid mode - ensure meta_id_to_shard has 1:1 mapping for backward compatibility
        // This allows existing load code to work correctly
        meta_id_to_shard.clear();
        meta_id_to_shard.reserve(num_shards);
        for (int i = 0; i < num_shards; i++) {
            meta_id_to_shard.push_back(i);  // meta_id[i] -> shard_id[i]
        }

        if (verbose) {
            printf("IndexKmeansShard: using centroid mode (meta_id_to_shard created with 1:1 mapping)\n");
        }
    }

    // Build meta-level index
    build_meta_index();

#ifdef PARTITION_NESTED
    // =========================================================================
    // KH: Add boundary vectors to assignments BEFORE building HNSW indices
    // This way, HNSW is built with all vectors (primary + secondary) from the start
    // =========================================================================
    std::vector<int> modified_assignments = assignments;  // Copy primary assignments
    std::vector<int> secondary_counts(num_shards, 0);

    if (!boundary_vectors.empty()) {
        printf("  [Preparing Boundary Vectors] Adding %zu boundary vectors to partitions...\n",
               boundary_vectors.size());
        fflush(stdout);

        // Find secondary shard for each boundary vector
        IndexFlatL2 centroid_index(d);
        centroid_index.add(num_shards, shard_centroids.data());

        constexpr idx_t PROGRESS_INTERVAL = 100000;
        idx_t next_progress = PROGRESS_INTERVAL;

        // Group boundary vectors by their secondary shard
        std::vector<std::vector<idx_t>> boundary_by_shard(num_shards);

        for (size_t i = 0; i < boundary_vectors.size(); i++) {
            idx_t vec_id = boundary_vectors[i];

            // Find the two nearest centroids
            std::vector<float> dists(2);
            std::vector<idx_t> labels(2);
            centroid_index.search(1, x + vec_id * d, 2, dists.data(), labels.data());

            int primary_shard = assignments[vec_id];
            int secondary_shard = static_cast<int>(labels[1]);  // Second nearest

            // If primary is already the second nearest (shouldn't happen), use first
            if (secondary_shard == primary_shard) {
                secondary_shard = static_cast<int>(labels[0]);
            }

            // Store this boundary vector for its secondary shard
            if (secondary_shard >= 0 && secondary_shard < num_shards &&
                secondary_shard != primary_shard) {
                boundary_by_shard[secondary_shard].push_back(vec_id);
                secondary_counts[secondary_shard]++;
            }

            // Progress reporting
            if (i > 0 && i >= next_progress) {
                printf("  [Preparing Boundary Vectors] Processed %zu / %zu vectors (%.1f%%)\n",
                       i, boundary_vectors.size(), 100.0 * i / boundary_vectors.size());
                fflush(stdout);
                next_progress += PROGRESS_INTERVAL;
            }
        }

        printf("  [Preparing Boundary Vectors] Summary:");
        for (int i = 0; i < num_shards; i++) {
            if (secondary_counts[i] > 0) {
                printf(" shard_%d=+%d", i, secondary_counts[i]);
            }
        }
        printf("\n");
        fflush(stdout);

        // Now we need to manually add boundary vectors to partition_to_global
        // BEFORE calling build_shard_indices, but build_shard_indices clears partition_to_global
        // So we'll need to modify the approach: extend the dataset temporarily

        // Create extended assignments that include boundary vector duplicates
        // Original approach won't work because build_shard_indices uses the assignments array index
        // Instead, we'll need to manually populate partition_to_global after build_shard_indices initializes it

        printf("  [Preparing Boundary Vectors] Will add boundary vectors during HNSW build\n");
        fflush(stdout);
    }
#endif  // PARTITION_NESTED

    // Build individual shard indices with primary assignments
    // We'll extend partition_to_global with boundary vectors inside build_shard_indices_with_boundaries
    build_shard_indices_with_boundaries(x, n, assignments, boundary_vectors);

#ifdef PARTITION_NESTED
    if (!boundary_vectors.empty() && verbose) {
        printf("IndexKmeansShard: final shard sizes (with boundary overlap):");
        for (int i = 0; i < num_shards; i++) {
            if (shard_indices[i]) {
                printf(" shard_%d=%zd", i, shard_indices[i]->ntotal);
            }
        }
        printf("\n");
    }
#endif  // PARTITION_NESTED

    ntotal = n;
    is_trained = true;

    if (verbose) {
        printf("IndexKmeansShard: training completed\n");
    }
}

std::vector<int> IndexKmeansShard::kmeans_partition(idx_t n, const float* x) {
    fprintf(stderr, "kmeans_partition: n=%zd, d=%d, num_shards=%d\n", n, d, num_shards);
    fflush(stderr);

    printf("  [K-means Clustering] Starting k-means with %d iterations for %d shards...\n",
           niter, num_shards);
    fflush(stdout);

    // =========================================================================
    // Common k-means clustering using FAISS Clustering (same for both modes)
    // =========================================================================
    ClusteringParameters cp;
    cp.niter = niter;
    cp.verbose = verbose;
    cp.update_index = false;

    fprintf(stderr, "Creating Clustering object with d=%d, k=%d\n", d, num_shards);
    fflush(stderr);

    Clustering clustering(d, num_shards, cp);

    fprintf(stderr, "Clustering object created successfully\n");
    fflush(stderr);

    // Create index for clustering based on metric type
    std::unique_ptr<Index> assign_index;
    if (metric_type == METRIC_L2) {
        assign_index.reset(new IndexFlatL2(d));
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        assign_index.reset(new IndexFlatIP(d));
    } else {
        FAISS_THROW_MSG("Unsupported metric type for IndexKmeansShard");
    }

    try {
        // Run K-means clustering
        auto start_time = std::chrono::high_resolution_clock::now();
        clustering.train(n, x, *assign_index);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        printf("  [K-means Clustering] Completed in %ld seconds\n", duration);
        fflush(stdout);

        // Get centroids from clustering result
        shard_centroids = clustering.centroids;
    } catch (const std::exception& e) {
        if (verbose) {
            printf("IndexKmeansShard: K-means failed: %s\n", e.what());
            printf("IndexKmeansShard: falling back to random initialization\n");
        }
        // Fallback to random initialization
        srand(42);
        shard_centroids.resize(num_shards * d);
        for (int i = 0; i < num_shards; i++) {
            idx_t random_idx = rand() % n;
            std::copy(x + random_idx * d, x + (random_idx + 1) * d,
                     shard_centroids.data() + i * d);
        }
    }

    // =========================================================================
    // Assign vectors to nearest centroids (OpenMP parallelized)
    // =========================================================================
    std::vector<int> assignments(n);

#ifdef PARTITION_NESTED
    // =========================================================================
    // PARTITION_NESTED: Distance-based boundary detection with OpenMP
    // Direct distance calculation to all centroids (parallelized)
    // =========================================================================
    int num_threads = omp_get_max_threads();
    if (verbose) {
        printf("IndexKmeansShard: detecting boundary vectors (ratio threshold=%.2f, threads=%d)\n",
               BOUNDARY_RATIO_THRESHOLD, num_threads);
    }

    // Thread-local boundary vectors to avoid lock contention
    std::vector<std::vector<idx_t>> thread_boundary_vectors(num_threads);

    // Progress tracking with atomic counter
    constexpr idx_t PROGRESS_INTERVAL = 100000;
    std::atomic<idx_t> progress_counter(0);
    idx_t last_reported = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid == 0 && verbose) {
            printf("Checking Boundary, The number of Threads: %d\n", num_threads);
        }
        std::vector<float> dists(num_shards);  // Distance to each centroid

        #pragma omp for schedule(static)
        for (idx_t i = 0; i < n; i++) {
            const float* vec = x + i * d;

            // Compute distance to all centroids
            for (int c = 0; c < num_shards; c++) {
                const float* centroid = shard_centroids.data() + c * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = vec[j] - centroid[j];
                    dist += diff * diff;
                }
                dists[c] = dist;
            }

            // Find 1st and 2nd nearest centroids
            int nearest = 0;
            int second_nearest = 1;
            if (dists[1] < dists[0]) {
                nearest = 1;
                second_nearest = 0;
            }
            for (int c = 2; c < num_shards; c++) {
                if (dists[c] < dists[nearest]) {
                    second_nearest = nearest;
                    nearest = c;
                } else if (dists[c] < dists[second_nearest]) {
                    second_nearest = c;
                }
            }

            // Assign to nearest centroid
            assignments[i] = nearest;

            // Check if boundary vector: d2/d1 < threshold
            float d1 = dists[nearest];
            float d2 = dists[second_nearest];
            if (d1 > 0 && (d2 / d1) < BOUNDARY_RATIO_THRESHOLD) {
                thread_boundary_vectors[tid].push_back(i);
            }

            // Progress reporting (thread 0 only, every PROGRESS_INTERVAL)
            if (tid == 0) {
                idx_t current = progress_counter.fetch_add(1, std::memory_order_relaxed);
                if (current - last_reported >= PROGRESS_INTERVAL) {
                    printf("  [Boundary Detection] Processed ~%zd / %zd vectors (%.1f%%)\n",
                           current, n, 100.0 * current / n);
                    fflush(stdout);
                    last_reported = current;
                }
            } else {
                progress_counter.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Merge thread-local boundary vectors
    boundary_vectors.clear();
    for (int t = 0; t < num_threads; t++) {
        boundary_vectors.insert(boundary_vectors.end(),
                                thread_boundary_vectors[t].begin(),
                                thread_boundary_vectors[t].end());
    }

    if (verbose) {
        printf("IndexKmeansShard: found %zu boundary vectors (%.2f%% of total)\n",
               boundary_vectors.size(),
               100.0 * boundary_vectors.size() / n);
    }

#else
    // =========================================================================
    // Standard mode: just assign to nearest centroid (OpenMP parallelized)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;

        // Find nearest centroid
        int nearest = 0;
        float min_dist = std::numeric_limits<float>::max();
        for (int c = 0; c < num_shards; c++) {
            const float* centroid = shard_centroids.data() + c * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = vec[j] - centroid[j];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                nearest = c;
            }
        }
        assignments[i] = nearest;
    }
    printf("[COMPLETE] Checking Boundary\n");
#endif  // PARTITION_NESTED

    // Log distribution
    if (verbose) {
        std::vector<int> counts(num_shards, 0);
        for (int shard_id : assignments) {
            if (shard_id >= 0 && shard_id < num_shards) {
                counts[shard_id]++;
            }
        }

        printf("IndexKmeansShard: shard distribution:");
        for (int i = 0; i < num_shards; i++) {
            printf(" shard_%d=%d", i, counts[i]);
        }
        printf("\n");
    }

    return assignments;
}

} // namespace faiss