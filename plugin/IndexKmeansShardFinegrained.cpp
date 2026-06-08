/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Cluster-merging strategy is selected at RUNTIME via `merge_mode`
// (MergeMode::PCA | GREEDY_CENTROID | GREEDY_MIN_DIST); see the header.

#include "IndexKmeansShardFinegrained.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>
#include <omp.h> // for parallel search
#include <algorithm>
#include <atomic>  // for progress tracking
#include <chrono>  // for timing
#include <cmath>   // for std::sqrt, std::abs
#include <fstream> // for file I/O
#include <limits>
#include <memory>
#include <utility> // for std::pair
// Router (meta) built from per-shard sub-clusters (VeX router); compile-time on.
#define SAMPLING_KMEANS
// Boundary replication is a RUNTIME toggle: base class `partition_nested` plus
// this class's `boundary_tau`. No compile-time macro.
namespace faiss {

IndexKmeansShardFinegrained::IndexKmeansShardFinegrained(
        int d,
        MetricType metric)
        : IndexShardBase(d, metric),
          niter(25),
          num_fine_clusters(500),
          partition_ratio(0.5f) {
    printf("IndexKmeansShardFinegrained constructor: d=%d, metric=%d\n",
           d,
           (int)metric);
}

void IndexKmeansShardFinegrained::train(idx_t n, const float* x) {
    fprintf(stderr,
            "IndexKmeansShardFinegrained::train called with n=%zd, d=%d, num_shards=%d, num_fine_clusters=%d\n",
            n,
            d,
            num_shards,
            num_fine_clusters);
    fflush(stderr);

    FAISS_THROW_IF_NOT_MSG(n > 0, "Cannot train on empty dataset");
    FAISS_THROW_IF_NOT_MSG(x, "Training data is null");
    FAISS_THROW_IF_NOT_MSG(
            num_shards > 0, "num_shards must be set before training");
    FAISS_THROW_IF_NOT_MSG(
            num_fine_clusters >= num_shards,
            "num_fine_clusters must be >= num_shards");

    const char* merge_name =
            (merge_mode == MergeMode::PCA) ? "PCA"
            : (merge_mode == MergeMode::GREEDY_CENTROID) ? "GREEDY_CENTROID"
                                                         : "GREEDY_MIN_DIST";
    printf("============================================================\n");
    printf("IndexKmeansShardFinegrained BUILD CONFIGURATION:\n");
    printf("============================================================\n");
    printf("  Merging mode:          %s\n", merge_name);
    printf("  Fine-grained clusters: %d\n", num_fine_clusters);
    printf("  Target shards:         %d\n", num_shards);
    printf("  Partition ratio:       %.2f (%.1f%% : %.1f%%)\n",
           partition_ratio,
           partition_ratio * 100,
           (1.0f - partition_ratio) * 100);
    printf("  K-means iterations:    %d\n", niter);
    printf("  Boundary replication:  %s (tau=%.4f)\n",
           partition_nested ? "ENABLED" : "DISABLED",
           boundary_tau);
    printf("  HNSW M:                %d\n", hnsw_M);
    printf("  HNSW efConstruction:   %d\n", ef_construction);
    printf("============================================================\n");

    if (verbose) {
        printf("IndexKmeansShardFinegrained: training on %zd vectors with %d fine clusters\n",
               n,
               num_fine_clusters);
    }

    // Step 1: Fine-grained K-means clustering
    std::vector<int> cluster_assignments = perform_finegrained_clustering(n, x);

    // Step 2: Compute cluster sizes
    std::vector<int> cluster_sizes(num_fine_clusters, 0);
    for (idx_t i = 0; i < n; i++) {
        cluster_sizes[cluster_assignments[i]]++;
    }

    // Step 3: Merge clusters into shards (runtime mode-dependent)
    std::vector<int> cluster_to_shard;
    switch (merge_mode) {
        case MergeMode::GREEDY_MIN_DIST:
            cluster_to_shard =
                    assign_clusters_to_shards_greedy(shard_centroids, cluster_sizes);
            break;
        case MergeMode::GREEDY_CENTROID:
            cluster_to_shard = assign_clusters_to_shards_greedy_centroid(
                    shard_centroids, cluster_sizes);
            break;
        case MergeMode::PCA:
        default:
            cluster_to_shard =
                    assign_clusters_to_shards_pca(shard_centroids, cluster_sizes);
            break;
    }

    // Step 4: Convert cluster assignments to shard assignments
    std::vector<int> shard_assignments(n);
    for (idx_t i = 0; i < n; i++) {
        int cluster_id = cluster_assignments[i];
        shard_assignments[i] = cluster_to_shard[cluster_id];
    }

    // Step 4.5: Save partitioned data to fvecs files if requested
    if (!partition_save_path.empty()) {
        printf("  [Partition Save] Saving partitioned data to fvecs files...\n");
        fflush(stdout);

        // Count vectors per shard
        std::vector<idx_t> shard_counts(num_shards, 0);
        for (idx_t i = 0; i < n; i++) {
            shard_counts[shard_assignments[i]]++;
        }

        // Create output files for each shard
        for (int shard_id = 0; shard_id < num_shards; shard_id++) {
            // Determine partition size ratio for filename
            float shard_ratio = (shard_id == 0) ? partition_ratio
                                                : (1.0f - partition_ratio);

            // Build filename: {base_name}_Partition_{ratio}.fvecs
            char ratio_str[16];
            snprintf(ratio_str, sizeof(ratio_str), "%.2f", shard_ratio);

            std::string filename = partition_save_path + "/" +
                    partition_base_name + "_Partition_" + ratio_str + ".fvecs";

            printf("  [Partition Save] Writing shard %d to %s (%zd vectors)...\n",
                   shard_id,
                   filename.c_str(),
                   shard_counts[shard_id]);
            fflush(stdout);

            std::ofstream ofs(filename, std::ios::binary);
            if (!ofs) {
                fprintf(stderr,
                        "ERROR: Cannot open file for writing: %s\n",
                        filename.c_str());
                continue;
            }

            // Write vectors belonging to this shard in fvecs format
            for (idx_t i = 0; i < n; i++) {
                if (shard_assignments[i] == shard_id) {
                    // fvecs format: [dim (int)] [float * dim]
                    ofs.write(reinterpret_cast<const char*>(&d), sizeof(int));
                    ofs.write(
                            reinterpret_cast<const char*>(x + i * d),
                            d * sizeof(float));
                }
            }

            ofs.close();
            printf("  [Partition Save] Shard %d saved successfully.\n",
                   shard_id);
        }

        printf("  [Partition Save] All partitions saved to %s\n",
               partition_save_path.c_str());
        fflush(stdout);
    }

    // Step 5: Setup meta_id_to_shard mapping for router graph
#ifndef SAMPLING_KMEANS
    meta_id_to_shard.clear();
    meta_id_to_shard.reserve(num_fine_clusters);
    for (int i = 0; i < num_fine_clusters; i++) {
        meta_id_to_shard.push_back(cluster_to_shard[i]);
    }

    if (verbose) {
        printf("IndexKmeansShardFinegrained: meta_id_to_shard mapping created (%zu entries)\n",
               meta_id_to_shard.size());
    }

    // Step 6: Set sampling mode to indicate fine-grained centroids
    // This ensures build_meta_index() uses all num_fine_clusters centroids
    num_samples_per_shard = num_fine_clusters / num_shards;
    printf("  [Meta Index] Building router graph with kmeans sampling\n");

    printf("  [Meta Index] Building router graph with %d fine-grained centroids...\n",
           num_fine_clusters);
    printf("  [Meta Index] Mode: FINE-GRAINED CENTROIDS (not traditional sampling)\n");
    printf("  [Meta Index] HNSW parameters: M=%d, efConstruction=%d\n",
           hnsw_M,
           ef_construction);
    fflush(stdout);
#else // SAMPLING_KMEANS - WJ
    printf("  [Meta Index] Building router graph with random sampling\n");
    compute_shard_centroids(x, n, shard_assignments);
#endif

    // Step 7: Build meta-level HNSW index from fine-grained centroids

    try {
        build_meta_index();
    } catch (const std::exception& e) {
        fprintf(stderr,
                "ERROR: build_meta_index() threw exception: %s\n",
                e.what());
        throw;
    }

    printf("  [Meta Index] Router graph built successfully! (%d centroids indexed)\n",
           num_fine_clusters);
    fflush(stdout);

    // WJ
    // =========================================================================
    // Compute final 2-shard centroids for boundary detection
    // =========================================================================
    printf("  [Computing Final Shard Centroids] Calculating centroids for %d shards...\n",
           num_shards);
    fflush(stdout);
    std::vector<float> final_shard_centroids(num_shards * d, 0.0f);
    std::vector<int> shard_counts(num_shards, 0);

    // Sum vectors by shard
    fprintf(stderr,
            "  [DEBUG-STDERR] Starting vector summation loop (n=%zd)\n",
            n);
    fprintf(stderr,
            "  [DEBUG-STDERR] shard_assignments.size()=%zu, expected=%zd\n",
            shard_assignments.size(),
            n);
    fprintf(stderr, "  [DEBUG-STDERR] x pointer=%p\n", (void*)x);
    fflush(stderr);

    idx_t progress_interval = n / 10;
    for (idx_t i = 0; i < n; i++) {
        // 진행 상황 출력
        if (i > 0 && i % progress_interval == 0) {
            fprintf(stderr,
                    "  [DEBUG-STDERR] Loop progress: %zd / %zd (%.1f%%)\n",
                    i,
                    n,
                    100.0 * i / n);
            fflush(stderr);
        }

        int shard_id = shard_assignments[i];
        if (shard_id >= 0 && shard_id < num_shards) {
            for (int j = 0; j < d; j++) {
                final_shard_centroids[shard_id * d + j] += x[i * d + j];
            }
            shard_counts[shard_id]++;
        }
    }

    // Compute averages
    fprintf(stderr, "  [DEBUG-STDERR] Computing averages for centroids\n");
    fflush(stderr);
    for (int shard_id = 0; shard_id < num_shards; shard_id++) {
        if (shard_counts[shard_id] > 0) {
            float inv_count = 1.0f / shard_counts[shard_id];
            for (int j = 0; j < d; j++) {
                final_shard_centroids[shard_id * d + j] *= inv_count;
            }
        }
    }
    fprintf(stderr, "  [DEBUG-STDERR] Averages computed successfully\n");
    fflush(stderr);

    printf("  [Computing Final Shard Centroids] Completed: shard_0=%d vectors, shard_1=%d vectors\n",
           shard_counts[0],
           shard_counts[1]);
    fflush(stdout);

    // =========================================================================
    // Boundary Detection (runtime toggle: partition_nested)
    // A vector is a boundary vector if dist_2nd / dist_1st < boundary_tau.
    // =========================================================================
    if (partition_nested) {
    printf("  [Boundary Detection] Detecting boundary vectors (tau=%.4f)...\n",
           boundary_tau);
    fflush(stdout);

    int num_threads = omp_get_max_threads();
    fprintf(stderr, "  [DEBUG-STDERR] num_threads = %d\n", num_threads);
    fflush(stderr);
    if (verbose) {
        printf("  [Boundary Detection] Using %d threads for parallel processing\n",
               num_threads);
    }

    // Thread-local boundary vectors to avoid lock contention
    fprintf(stderr,
            "  [DEBUG-STDERR] Allocating thread_boundary_vectors for %d threads\n",
            num_threads);
    fflush(stderr);
    std::vector<std::vector<idx_t>> thread_boundary_vectors(num_threads);
    fprintf(stderr,
            "  [DEBUG-STDERR] thread_boundary_vectors allocated successfully\n");
    fflush(stderr);

    // Progress tracking with atomic counter
    // constexpr idx_t PROGRESS_INTERVAL = 100000;
    // std::atomic<idx_t> progress_counter(0);
    // std::atomic<idx_t> last_reported(0);
    // idx_t last_reported = 0;

    std::atomic<idx_t> bd_progress_counter(0);
    idx_t bd_progress_interval = n / 10;

    fprintf(stderr, "  [DEBUG-STDERR] About to enter OpenMP parallel region\n");
    fflush(stderr);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        // 스레드 0만 출력 wj
        if (tid == 0) {
            fprintf(stderr,
                    "  [DEBUG-STDERR] Inside parallel region, total_threads=%d\n",
                    total_threads);
            fflush(stderr);
        }

        std::vector<float> dists(num_shards); // Distance to each shard centroid

#pragma omp for schedule(static)
        for (idx_t i = 0; i < n; i++) {
            const float* vec = x + i * d;

            // Compute distance to all shard centroids
            for (int c = 0; c < num_shards; c++) {
                const float* centroid = final_shard_centroids.data() + c * d;
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

            // Assign to nearest centroid (should match
            // shard_assignments[i]) Note: This is just a sanity check,
            // actual assignment already done

            // Check if boundary vector: d2/d1 < threshold
            float d1 = dists[nearest];
            float d2 = dists[second_nearest];
            if (d1 > 0 && (d2 / d1) < boundary_tau) {
                thread_boundary_vectors[tid].push_back(i);
            }

            // 진행 상황 출력 wj
            idx_t bd_current =
                    bd_progress_counter.fetch_add(1, std::memory_order_relaxed);
            if (tid == 0 && bd_current > 0 &&
                bd_current % bd_progress_interval == 0) {
                fprintf(stderr,
                        "  [DEBUG-STDERR] Boundary Detection: %zd / %zd (%.1f%%)\n",
                        bd_current,
                        n,
                        100.0 * bd_current / n);
                fflush(stderr);
            }

            // Progress reporting (thread 0 only, every PROGRESS_INTERVAL)
            // if (tid == 0) {
            //     idx_t current = progress_counter.fetch_add(
            //             1, std::memory_order_relaxed);
            //     if (current - last_reported >= PROGRESS_INTERVAL) {
            //         printf("  [Boundary Detection] Processed ~%zd / %zd
            //         vectors (%.1f%%)\n",
            //                current,
            //                n,
            //                100.0 * current / n);
            //         fflush(stdout);
            //         last_reported = current;
            //     }
            // } else {
            //     progress_counter.fetch_add(1, std::memory_order_relaxed);
            // }
        }
    }

    // Merge thread-local boundary vectors
    boundary_vectors.clear();
    for (int t = 0; t < num_threads; t++) {
        // fprintf(stderr, "  [DEBUG-STDERR] Merging from thread %d: %zu
        // vectors\n",
        //         t, thread_boundary_vectors[t].size());
        boundary_vectors.insert(
                boundary_vectors.end(),
                thread_boundary_vectors[t].begin(),
                thread_boundary_vectors[t].end());
    }
    fprintf(stderr,
            "  [DEBUG-STDERR] Total boundary vectors: %zu\n",
            boundary_vectors.size());
    fflush(stderr);

    if (verbose) {
        printf("  [Boundary Detection] Found %zu boundary vectors (%.2f%% of total)\n",
               boundary_vectors.size(),
               100.0 * boundary_vectors.size() / n);
        fflush(stdout);
    }

    // WJ: 이 블록은 shard_indices가 아직 빌드되지 않았으므로 제거 또는 이동
    // 필요 if (!boundary_vectors.empty() && verbose) {
    //     printf("IndexKmeansShardFinegrained: final shard sizes (with boundary
    //     overlap):"); for (int i = 0; i < num_shards; i++) {
    //         if (shard_indices[i]) {
    //             printf(" shard_%d=%zd", i, shard_indices[i]->ntotal);
    //         }
    //     }
    //     printf("\n");
    // }
    fprintf(stderr, "  [DEBUG-STDERR] Boundary Detection section completed\n");
    fflush(stderr);
    }  // end if (partition_nested)

    fprintf(stderr, "  [DEBUG-STDERR] Saving fine centroids and swapping\n");
    fflush(stderr);
    std::vector<float> saved_fine_centroids = shard_centroids;
    shard_centroids = final_shard_centroids;
    fprintf(stderr, "  [DEBUG-STDERR] Centroid swap done\n");
    fflush(stderr);

    // WJ

    // Step 8: Build individual shard HNSW indices
    fprintf(stderr,
            "  [DEBUG-STDERR] About to call build_shard_indices_with_boundaries\n");
    fprintf(stderr,
            "  [DEBUG-STDERR]   x=%p, n=%zd, shard_assignments.size()=%zu, boundary_vectors.size()=%zu\n",
            (void*)x,
            n,
            shard_assignments.size(),
            boundary_vectors.size());
    fflush(stderr);
    build_shard_indices_with_boundaries(
            x, n, shard_assignments, boundary_vectors);
    fprintf(stderr,
            "  [DEBUG-STDERR] build_shard_indices_with_boundaries completed\n");
    fflush(stderr);
    shard_centroids = saved_fine_centroids;
    // Log final distribution
    if (verbose) {
        std::vector<int> shard_counts(num_shards, 0);
        for (int shard_id : shard_assignments) {
            shard_counts[shard_id]++;
        }
        printf("IndexKmeansShardFinegrained: final shard distribution:");
        for (int i = 0; i < num_shards; i++) {
            printf(" shard_%d=%d (%.2f%%)",
                   i,
                   shard_counts[i],
                   100.0 * shard_counts[i] / n);
        }
        printf("\n");
    }

    ntotal = n;
    is_trained = true;

    if (verbose) {
        printf("IndexKmeansShardFinegrained: training completed\n");
    }
}

std::vector<int> IndexKmeansShardFinegrained::perform_finegrained_clustering(
        idx_t n,
        const float* x) {
    printf("  [Fine-grained K-means] Starting clustering with k=%d, niter=%d...\n",
           num_fine_clusters,
           niter);
    fflush(stdout);

    // Setup clustering parameters
    ClusteringParameters cp;
    cp.niter = niter;
    cp.verbose = verbose;
    cp.update_index = false;

    // Create clustering object
    Clustering clustering(d, num_fine_clusters, cp);

    // Create index for clustering based on metric type
    std::unique_ptr<Index> assign_index;
    if (metric_type == METRIC_L2) {
        assign_index.reset(new IndexFlatL2(d));
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        assign_index.reset(new IndexFlatIP(d));
    } else {
        FAISS_THROW_MSG(
                "Unsupported metric type for IndexKmeansShardFinegrained");
    }

    try {
        // Run K-means clustering
        auto start_time = std::chrono::high_resolution_clock::now();
        clustering.train(n, x, *assign_index);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                                end_time - start_time)
                                .count();
        printf("  [Fine-grained K-means] Completed in %ld seconds\n", duration);
        fflush(stdout);

        // Store centroids for router graph
        shard_centroids = clustering.centroids;
    } catch (const std::exception& e) {
        if (verbose) {
            printf("IndexKmeansShardFinegrained: K-means failed: %s\n",
                   e.what());
        }
        FAISS_THROW_MSG("Fine-grained K-means clustering failed");
    }

    // Assign each vector to nearest centroid
    printf("  [Fine-grained K-means] Assigning vectors to clusters...\n");
    fflush(stdout);

    std::vector<int> assignments(n);
    std::vector<idx_t> labels(n);
    std::vector<float> distances(n);

    // Add centroids to index for nearest neighbor search
    assign_index->reset();
    assign_index->add(num_fine_clusters, shard_centroids.data());

    // Search for nearest centroid for each vector
    assign_index->search(n, x, 1, distances.data(), labels.data());

    for (idx_t i = 0; i < n; i++) {
        assignments[i] = static_cast<int>(labels[i]);
    }

    // Log cluster size distribution
    if (verbose) {
        std::vector<int> counts(num_fine_clusters, 0);
        for (int cluster_id : assignments) {
            if (cluster_id >= 0 && cluster_id < num_fine_clusters) {
                counts[cluster_id]++;
            }
        }

        int min_size = *std::min_element(counts.begin(), counts.end());
        int max_size = *std::max_element(counts.begin(), counts.end());
        int avg_size = n / num_fine_clusters;

        printf("  [Fine-grained K-means] Cluster sizes: min=%d, max=%d, avg=%d\n",
               min_size,
               max_size,
               avg_size);
    }

    return assignments;
}

std::vector<int> IndexKmeansShardFinegrained::assign_clusters_to_shards_greedy(
        const std::vector<float>& centroids,
        const std::vector<int>& cluster_sizes) {
    printf("  [Greedy Merging] Assigning %d clusters to %d shards...\n",
           num_fine_clusters,
           num_shards);
    fflush(stdout);

    FAISS_THROW_IF_NOT_MSG(
            num_shards == 2, "Greedy merging currently only supports 2 shards");

    // Step 1: Find two farthest centroids as initial seeds
    printf("  [Greedy Merging] Finding initial seeds...\n");
    int seed0 = 0, seed1 = 0;
    float max_dist = 0;

    for (int i = 0; i < num_fine_clusters; i++) {
        for (int j = i + 1; j < num_fine_clusters; j++) {
            float dist = compute_centroid_distance(
                    centroids.data() + i * d, centroids.data() + j * d);
            if (dist > max_dist) {
                max_dist = dist;
                seed0 = i;
                seed1 = j;
            }
        }
    }

    printf("  [Greedy Merging] Seeds: cluster_%d and cluster_%d (distance=%.2f)\n",
           seed0,
           seed1,
           max_dist);

    // Initialize cluster-to-shard mapping
    std::vector<int> cluster_to_shard(num_fine_clusters, -1);
    cluster_to_shard[seed0] = 0;
    cluster_to_shard[seed1] = 1;

    // Initialize shard sizes
    std::vector<int> shard_sizes(num_shards, 0);
    shard_sizes[0] = cluster_sizes[seed0];
    shard_sizes[1] = cluster_sizes[seed1];

    // Calculate total vectors and target size
    idx_t total_vectors = 0;
    for (int size : cluster_sizes) {
        total_vectors += size;
    }
    idx_t target_size0 = static_cast<idx_t>(total_vectors * partition_ratio);

    printf("  [Greedy Merging] Target distribution: shard_0=%zd (%.1f%%), shard_1=%zd (%.1f%%)\n",
           target_size0,
           partition_ratio * 100,
           total_vectors - target_size0,
           (1.0f - partition_ratio) * 100);

    // Step 2: Greedy assignment of remaining clusters
    int assigned_count = 2; // seed0 and seed1
    while (assigned_count < num_fine_clusters) {
        int best_cluster = -1;
        int best_shard = -1;
        float best_distance = std::numeric_limits<float>::max();

        // Find the best (cluster, shard) pair
        for (int c = 0; c < num_fine_clusters; c++) {
            if (cluster_to_shard[c] != -1)
                continue; // Already assigned

            // Calculate distance to each shard
            float dist_to_0 =
                    min_distance_to_shard(c, 0, cluster_to_shard, centroids);
            float dist_to_1 =
                    min_distance_to_shard(c, 1, cluster_to_shard, centroids);

            // Prefer closer shard
            int preferred_shard = (dist_to_0 < dist_to_1) ? 0 : 1;
            float preferred_dist =
                    (preferred_shard == 0) ? dist_to_0 : dist_to_1;

            // Check size constraints (allow 10% overflow)
            if (preferred_shard == 0) {
                if (shard_sizes[0] + cluster_sizes[c] > target_size0 * 1.06) {
                    preferred_shard = 1;
                    preferred_dist = dist_to_1;
                }
            } else {
                if (shard_sizes[1] + cluster_sizes[c] >
                    (total_vectors - target_size0) * 1.06) {
                    preferred_shard = 0;
                    preferred_dist = dist_to_0;
                }
            }

            // Update best choice
            if (preferred_dist < best_distance) {
                best_distance = preferred_dist;
                best_cluster = c;
                best_shard = preferred_shard;
            }
        }

        // Assign best cluster to best shard
        if (best_cluster == -1) {
            fprintf(stderr, "Warning: Could not find unassigned cluster!\n");
            break;
        }

        cluster_to_shard[best_cluster] = best_shard;
        shard_sizes[best_shard] += cluster_sizes[best_cluster];
        assigned_count++;

        // Progress reporting every 10%
        if (assigned_count % (num_fine_clusters / 10) == 0) {
            printf("  [Greedy Merging] Progress: %d / %d clusters assigned (%.1f%%)\n",
                   assigned_count,
                   num_fine_clusters,
                   100.0 * assigned_count / num_fine_clusters);
        }
    }

    // Final distribution report
    printf("  [Greedy Merging] Final distribution: shard_0=%d (%.1f%%), shard_1=%d (%.1f%%)\n",
           shard_sizes[0],
           100.0 * shard_sizes[0] / total_vectors,
           shard_sizes[1],
           100.0 * shard_sizes[1] / total_vectors);
    printf("  [Greedy Merging] Target vs Actual: shard_0 target=%zd actual=%d (diff=%+d)\n",
           target_size0,
           shard_sizes[0],
           shard_sizes[0] - (int)target_size0);

    return cluster_to_shard;
}

float IndexKmeansShardFinegrained::compute_centroid_distance(
        const float* centroid1,
        const float* centroid2) const {
    float dist = 0.0f;
    for (int k = 0; k < d; k++) {
        float diff = centroid1[k] - centroid2[k];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

float IndexKmeansShardFinegrained::min_distance_to_shard(
        int cluster_id,
        int shard_id,
        const std::vector<int>& cluster_to_shard,
        const std::vector<float>& centroids) const {
    float min_dist = std::numeric_limits<float>::max();
    const float* target_centroid = centroids.data() + cluster_id * d;

    for (int i = 0; i < num_fine_clusters; i++) {
        if (cluster_to_shard[i] == shard_id) {
            const float* shard_centroid = centroids.data() + i * d;
            float dist =
                    compute_centroid_distance(target_centroid, shard_centroid);
            min_dist = std::min(min_dist, dist);
        }
    }

    return min_dist;
}

// ============================================================================
// Helper functions for GREEDY_CENTROID mode
// ============================================================================

void IndexKmeansShardFinegrained::compute_partition_centroid(
        int shard_id,
        const std::vector<int>& cluster_to_shard,
        const std::vector<float>& centroids,
        const std::vector<int>& cluster_sizes,
        float* partition_centroid) const {
    // Initialize to zero
    for (int j = 0; j < d; j++) {
        partition_centroid[j] = 0.0f;
    }

    // Weighted sum of cluster centroids (weighted by cluster size)
    int total_vectors = 0;
    for (int i = 0; i < num_fine_clusters; i++) {
        if (cluster_to_shard[i] == shard_id) {
            int weight = cluster_sizes[i];
            const float* cluster_centroid = centroids.data() + i * d;
            for (int j = 0; j < d; j++) {
                partition_centroid[j] += cluster_centroid[j] * weight;
            }
            total_vectors += weight;
        }
    }

    // Compute average
    if (total_vectors > 0) {
        float inv_total = 1.0f / total_vectors;
        for (int j = 0; j < d; j++) {
            partition_centroid[j] *= inv_total;
        }
    }
}

float IndexKmeansShardFinegrained::distance_to_partition_centroid(
        int cluster_id,
        const std::vector<float>& centroids,
        const float* partition_centroid) const {
    const float* cluster_centroid = centroids.data() + cluster_id * d;
    return compute_centroid_distance(cluster_centroid, partition_centroid);
}

// ============================================================================
// MERGING_MODE_GREEDY_CENTROID implementation
// ============================================================================

std::vector<int> IndexKmeansShardFinegrained::
        assign_clusters_to_shards_greedy_centroid(
                const std::vector<float>& centroids,
                const std::vector<int>& cluster_sizes) {
    printf("  [Greedy Centroid Merging] Assigning %d clusters to %d shards...\n",
           num_fine_clusters,
           num_shards);
    fflush(stdout);

    FAISS_THROW_IF_NOT_MSG(
            num_shards == 2,
            "Greedy centroid merging currently only supports 2 shards");

    // Step 1: Find two farthest centroids as initial seeds
    printf("  [Greedy Centroid Merging] Finding initial seeds...\n");
    int seed0 = 0, seed1 = 0;
    float max_dist = 0;

    for (int i = 0; i < num_fine_clusters; i++) {
        for (int j = i + 1; j < num_fine_clusters; j++) {
            float dist = compute_centroid_distance(
                    centroids.data() + i * d, centroids.data() + j * d);
            if (dist > max_dist) {
                max_dist = dist;
                seed0 = i;
                seed1 = j;
            }
        }
    }

    printf("  [Greedy Centroid Merging] Seeds: cluster_%d and cluster_%d (distance=%.2f)\n",
           seed0,
           seed1,
           max_dist);

    // Initialize cluster-to-shard mapping
    std::vector<int> cluster_to_shard(num_fine_clusters, -1);
    cluster_to_shard[seed0] = 0;
    cluster_to_shard[seed1] = 1;

    // Initialize shard sizes
    std::vector<int> shard_sizes(num_shards, 0);
    shard_sizes[0] = cluster_sizes[seed0];
    shard_sizes[1] = cluster_sizes[seed1];

    // Calculate total vectors and target size
    idx_t total_vectors = 0;
    for (int size : cluster_sizes) {
        total_vectors += size;
    }
    idx_t target_size0 = static_cast<idx_t>(total_vectors * partition_ratio);

    printf("  [Greedy Centroid Merging] Target distribution: shard_0=%zd (%.1f%%), shard_1=%zd (%.1f%%)\n",
           target_size0,
           partition_ratio * 100,
           total_vectors - target_size0,
           (1.0f - partition_ratio) * 100);

    // Initialize partition centroids
    std::vector<float> partition_centroid_0(d);
    std::vector<float> partition_centroid_1(d);

    // Step 2: Greedy assignment of remaining clusters
    int assigned_count = 2; // seed0 and seed1
    while (assigned_count < num_fine_clusters) {
        // Update partition centroids before each assignment
        compute_partition_centroid(
                0,
                cluster_to_shard,
                centroids,
                cluster_sizes,
                partition_centroid_0.data());
        compute_partition_centroid(
                1,
                cluster_to_shard,
                centroids,
                cluster_sizes,
                partition_centroid_1.data());

        int best_cluster = -1;
        int best_shard = -1;
        float best_distance = std::numeric_limits<float>::max();

        // Find the best (cluster, shard) pair
        for (int c = 0; c < num_fine_clusters; c++) {
            if (cluster_to_shard[c] != -1)
                continue; // Already assigned

            // Calculate distance to each partition centroid
            float dist_to_0 = distance_to_partition_centroid(
                    c, centroids, partition_centroid_0.data());
            float dist_to_1 = distance_to_partition_centroid(
                    c, centroids, partition_centroid_1.data());

            // Prefer closer shard
            int preferred_shard = (dist_to_0 < dist_to_1) ? 0 : 1;
            float preferred_dist =
                    (preferred_shard == 0) ? dist_to_0 : dist_to_1;

            // Check size constraints (allow 6% overflow)
            if (preferred_shard == 0) {
                if (shard_sizes[0] + cluster_sizes[c] > target_size0 * 1.06) {
                    preferred_shard = 1;
                    preferred_dist = dist_to_1;
                }
            } else {
                if (shard_sizes[1] + cluster_sizes[c] >
                    (total_vectors - target_size0) * 1.06) {
                    preferred_shard = 0;
                    preferred_dist = dist_to_0;
                }
            }

            // Update best choice
            if (preferred_dist < best_distance) {
                best_distance = preferred_dist;
                best_cluster = c;
                best_shard = preferred_shard;
            }
        }

        // Assign best cluster to best shard
        if (best_cluster == -1) {
            fprintf(stderr, "Warning: Could not find unassigned cluster!\n");
            break;
        }

        cluster_to_shard[best_cluster] = best_shard;
        shard_sizes[best_shard] += cluster_sizes[best_cluster];
        assigned_count++;

        // Progress reporting every 10%
        if (assigned_count % (num_fine_clusters / 10) == 0) {
            printf("  [Greedy Centroid Merging] Progress: %d / %d clusters assigned (%.1f%%)\n",
                   assigned_count,
                   num_fine_clusters,
                   100.0 * assigned_count / num_fine_clusters);
        }
    }

    // Final distribution report
    printf("  [Greedy Centroid Merging] Final distribution: shard_0=%d (%.1f%%), shard_1=%d (%.1f%%)\n",
           shard_sizes[0],
           100.0 * shard_sizes[0] / total_vectors,
           shard_sizes[1],
           100.0 * shard_sizes[1] / total_vectors);
    printf("  [Greedy Centroid Merging] Target vs Actual: shard_0 target=%zd actual=%d (diff=%+d)\n",
           target_size0,
           shard_sizes[0],
           shard_sizes[0] - (int)target_size0);

    return cluster_to_shard;
}

// ============================================================================
// MERGING_MODE_PCA implementation
// ============================================================================

std::vector<int> IndexKmeansShardFinegrained::assign_clusters_to_shards_pca(
        const std::vector<float>& centroids,
        const std::vector<int>& cluster_sizes) {
    printf("  [PCA Bisection] Assigning %d clusters to %d shards...\n",
           num_fine_clusters,
           num_shards);
    fflush(stdout);

    FAISS_THROW_IF_NOT_MSG(
            num_shards == 2, "PCA bisection currently only supports 2 shards");

    // Step 1: Compute mean of all cluster centroids (weighted by cluster size)
    printf("  [PCA Bisection] Computing weighted mean...\n");
    std::vector<float> mean(d, 0.0f);
    idx_t total_vectors = 0;

    for (int i = 0; i < num_fine_clusters; i++) {
        int weight = cluster_sizes[i];
        for (int j = 0; j < d; j++) {
            mean[j] += centroids[i * d + j] * weight;
        }
        total_vectors += weight;
    }
    for (int j = 0; j < d; j++) {
        mean[j] /= total_vectors;
    }

    // Step 2: Compute centered data (subtract mean)
    printf("  [PCA Bisection] Centering data...\n");
    std::vector<float> centered(num_fine_clusters * d);
    for (int i = 0; i < num_fine_clusters; i++) {
        for (int j = 0; j < d; j++) {
            centered[i * d + j] = centroids[i * d + j] - mean[j];
        }
    }

    // Step 3: Find first principal component using Power Iteration
    // PC1 = direction of maximum variance
    printf("  [PCA Bisection] Computing first principal component (power iteration)...\n");

    std::vector<float> pc1(d);
    // Initialize with random-ish vector (use first centered point)
    for (int j = 0; j < d; j++) {
        pc1[j] = centered[j]; // Use first point as initial guess
    }

    // Normalize initial vector
    float norm = 0.0f;
    for (int j = 0; j < d; j++) {
        norm += pc1[j] * pc1[j];
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (int j = 0; j < d; j++) {
            pc1[j] /= norm;
        }
    }

    // Power iteration: v = Cov * v / ||Cov * v||
    // Cov * v = (1/n) * X^T * X * v = (1/n) * X^T * (X * v)
    const int max_iterations = 100;
    const float convergence_threshold = 1e-6f;

    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<float> pc1_new(d, 0.0f);

        // Step 3a: Compute X * v (project each point onto pc1)
        std::vector<float> Xv(num_fine_clusters, 0.0f);
        for (int i = 0; i < num_fine_clusters; i++) {
            float weight = std::sqrt(
                    (float)cluster_sizes[i]); // Weight by sqrt of cluster size
            for (int j = 0; j < d; j++) {
                Xv[i] += centered[i * d + j] * pc1[j];
            }
            Xv[i] *= weight;
        }

        // Step 3b: Compute X^T * (X * v)
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < num_fine_clusters; i++) {
                float weight = std::sqrt((float)cluster_sizes[i]);
                pc1_new[j] += centered[i * d + j] * Xv[i] * weight;
            }
        }

        // Step 3c: Normalize
        norm = 0.0f;
        for (int j = 0; j < d; j++) {
            norm += pc1_new[j] * pc1_new[j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (int j = 0; j < d; j++) {
                pc1_new[j] /= norm;
            }
        }

        // Check convergence (dot product close to 1 means vectors are aligned)
        float dot = 0.0f;
        for (int j = 0; j < d; j++) {
            dot += pc1[j] * pc1_new[j];
        }

        pc1 = pc1_new;

        if (std::abs(std::abs(dot) - 1.0f) < convergence_threshold) {
            printf("  [PCA Bisection] Power iteration converged at iteration %d\n",
                   iter + 1);
            break;
        }
    }

    // Step 4: Project all cluster centroids onto PC1
    printf("  [PCA Bisection] Projecting clusters onto PC1...\n");
    std::vector<std::pair<float, int>> projections(num_fine_clusters);
    for (int i = 0; i < num_fine_clusters; i++) {
        float proj = 0.0f;
        for (int j = 0; j < d; j++) {
            proj += centered[i * d + j] * pc1[j];
        }
        projections[i] = {proj, i};
    }

    // Step 5: Sort by projection value
    std::sort(projections.begin(), projections.end());

    // Step 6: Find split point that achieves target partition ratio
    printf("  [PCA Bisection] Finding optimal split point for %.1f%% : %.1f%% ratio...\n",
           partition_ratio * 100,
           (1.0f - partition_ratio) * 100);

    idx_t target_size0 = static_cast<idx_t>(total_vectors * partition_ratio);
    idx_t cumulative_size = 0;
    int split_index = 0;

    for (int i = 0; i < num_fine_clusters; i++) {
        int cluster_id = projections[i].second;
        cumulative_size += cluster_sizes[cluster_id];

        if (cumulative_size >= target_size0) {
            // Check if current or previous split is closer to target
            idx_t diff_current = std::abs(
                    (long long)cumulative_size - (long long)target_size0);
            idx_t prev_cumulative = cumulative_size - cluster_sizes[cluster_id];
            idx_t diff_prev = std::abs(
                    (long long)prev_cumulative - (long long)target_size0);

            if (diff_prev < diff_current && i > 0) {
                split_index = i; // Split before current cluster
            } else {
                split_index = i + 1; // Split after current cluster
            }
            break;
        }
    }

    // Handle edge case
    if (split_index == 0)
        split_index = 1;
    if (split_index >= num_fine_clusters)
        split_index = num_fine_clusters - 1;

    // Step 7: Assign clusters to shards based on split
    std::vector<int> cluster_to_shard(num_fine_clusters);
    std::vector<int> shard_sizes(num_shards, 0);

    for (int i = 0; i < num_fine_clusters; i++) {
        int cluster_id = projections[i].second;
        int shard = (i < split_index) ? 0 : 1;
        cluster_to_shard[cluster_id] = shard;
        shard_sizes[shard] += cluster_sizes[cluster_id];
    }

    // Report results
    printf("  [PCA Bisection] Split at projection index %d (of %d)\n",
           split_index,
           num_fine_clusters);
    printf("  [PCA Bisection] Projection range: [%.4f, %.4f]\n",
           projections[0].first,
           projections[num_fine_clusters - 1].first);
    printf("  [PCA Bisection] Final distribution: shard_0=%d (%.1f%%), shard_1=%d (%.1f%%)\n",
           shard_sizes[0],
           100.0 * shard_sizes[0] / total_vectors,
           shard_sizes[1],
           100.0 * shard_sizes[1] / total_vectors);
    printf("  [PCA Bisection] Target vs Actual: shard_0 target=%zd actual=%d (diff=%+d)\n",
           target_size0,
           shard_sizes[0],
           shard_sizes[0] - (int)target_size0);

    return cluster_to_shard;
}

} // namespace faiss
