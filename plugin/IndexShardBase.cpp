/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "IndexShardBase.h"

#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>
#include <random>  // KH: for random sampling mode
#include <set>     // KH: for unique shard selection in route_query
#include <fstream> // KH: for CSV logging of top-k results

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/Clustering.h>  // KH: for sub-clustering in SAMPLING_KMEANS mode
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>

// ============================================================================
// KH: Mode Configuration - Uncomment to enable each mode
// ============================================================================

// ----- Sampling Mode -----
// Router (meta) index is built from per-shard sub-cluster centroids. This is
// the VeX router and is kept compile-time on.
#define SAMPLING_KMEANS

// ----- Partition Mode (boundary vector replication) -----
// NOTE: this is now a RUNTIME toggle, IndexShardBase::partition_nested, set via
// the build driver (-nested on|off). The old PARTITION_NESTED macro is removed.

// ============================================================================
// KH: Tunable Parameters - Modify these values to adjust behavior
// ============================================================================

// Number of sub-clusters per shard (used when SAMPLING_KMEANS is enabled)
// Each shard will have this many centroids in the meta index
constexpr int SUB_CLUSTERS_PER_SHARD = 500;

// Number of k-means iterations for sub-clustering (used when SAMPLING_KMEANS is enabled)
constexpr int SUB_CLUSTERING_NITER = 10;

// Boundary ratio threshold (used when PARTITION_NESTED is enabled)
// A vector is a boundary if: dist_to_2nd_nearest / dist_to_1st_nearest < threshold
// Lower value = more selective (fewer boundary vectors)
// Note: Actual boundary detection logic is in IndexKmeansShard.cpp
constexpr float BOUNDARY_RATIO_THRESHOLD = 1.01f;  // 1% threshold

// ============================================================================

#ifdef FAISS_WITH_DOCA
#include "pack.h"  // For htonq/ntohq byte order conversion
DOCA_LOG_REGISTER(FAISS_SHARD);
#define SERVER_NAME "dma copy server"
#endif

namespace faiss {

IndexShardBase::IndexShardBase(int d, MetricType metric)
        : Index(d, metric) {
    is_trained = false;
#ifdef FAISS_WITH_DOCA
    std::memset(&dma_cfg, 0, sizeof(dma_cfg));
    doca_initialized = false;
#endif
}

IndexShardBase::~IndexShardBase() {
#ifdef FAISS_WITH_DOCA
    close_doca_comch();
#endif
    delete meta_index;
    for (auto* idx : shard_indices) {
        delete idx;
    }
}

void IndexShardBase::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}

void IndexShardBase::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    if (!is_trained) {
        // First add: auto-train using all provided data
        if (verbose) {
            printf("IndexShard: auto-training on %zd vectors\n", n);
        }
        train(n, x);
    } else {
        // Already trained: add vectors to existing shards
        if (verbose) {
            printf("IndexShard: adding %zd vectors to existing shards\n", n);
        }
        add_to_existing_shards(n, x, xids);
    }
}

void IndexShardBase::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index not trained");

    // Handle search parameters
    int effective_nprobe = nprobe;
    if (params_in) {
        const SearchParametersShard* params = 
            dynamic_cast<const SearchParametersShard*>(params_in);
        if (params) {
            effective_nprobe = params->nprobe;
        }
    }
    effective_nprobe = std::min(effective_nprobe, num_shards);
    
    if (verbose) {
        printf("IndexShard search: %zd queries, k=%zd, nprobe=%d\n",
               n, k, effective_nprobe);
    }

    // Use batch processing for better efficiency
    search_batch(n, x, k, distances, labels, effective_nprobe);
}

void IndexShardBase::reset() {
    delete meta_index;
    meta_index = nullptr;
    for (auto* idx : shard_indices) {
        delete idx;
    }
    shard_indices.clear();
    partition_to_global.clear();
    vector_to_shard.clear();
    shard_centroids.clear();
    ntotal = 0;
    is_trained = false;
}

void IndexShardBase::build_meta_index() {
    FAISS_THROW_IF_NOT_MSG(!shard_centroids.empty(), "Shard centroids not computed");

    // KH: Determine number of vectors based on mode (sampling vs centroid)
    size_t num_meta_vectors = (num_samples_per_shard > 0) ?
                              meta_id_to_shard.size() :
                              num_shards;

    if (verbose) {
        if (num_samples_per_shard > 0) {
            printf("IndexShard: building meta index with %zu sampled vectors (using HNSW)\n",
                   num_meta_vectors);
        } else {
            printf("IndexShard: building meta index with %d centroids (using HNSW)\n",
                   num_shards);
        }
    }

    // Use HNSW for meta-level routing (faster search for large meta index)
    delete meta_index;
    IndexHNSWFlat* meta_hnsw = new IndexHNSWFlat(d, hnsw_M, metric_type);
    meta_hnsw->hnsw.efConstruction = ef_construction;
    meta_hnsw->hnsw.efSearch = ef_search;
    meta_hnsw->verbose = verbose;
    meta_index = meta_hnsw;

    // KH: Add vectors to meta index (centroids or sampled vectors)
    meta_index->add(num_meta_vectors, shard_centroids.data());
}

void IndexShardBase::build_shard_indices(const float* x, idx_t n,
                                        const std::vector<int>& assignments) {
    if (verbose) {
        printf("IndexShard: building %d shard indices\n", num_shards);
    }

    // Initialize shard structures
    for (auto* idx : shard_indices) {
        delete idx;
    }
    shard_indices.clear();
    shard_indices.resize(num_shards, nullptr);
    partition_to_global.resize(num_shards);
    vector_to_shard = assignments;

    // Group vectors by shard with progress tracking
    printf("  [Grouping Vectors] Starting assignment to %d shards...\n", num_shards);
    fflush(stdout);
    constexpr idx_t PROGRESS_INTERVAL = 100000;
    idx_t next_progress = PROGRESS_INTERVAL;

    for (idx_t i = 0; i < n; i++) {
        int shard_id = assignments[i];
        if (shard_id >= 0 && shard_id < num_shards) {
            partition_to_global[shard_id].push_back(i);
        }

        // Progress reporting
        if (i >= next_progress) {
            printf("  [Grouping Vectors] Processed %zd / %zd vectors (%.1f%%)\n",
                   i, n, 100.0 * i / n);
            fflush(stdout);
            next_progress += PROGRESS_INTERVAL;
        }
    }
    printf("  [Grouping Vectors] Complete: %zd vectors assigned\n", n);
    fflush(stdout);

    // Build HNSW for each shard
    printf("  [Building HNSW] Starting to build %d shard indices...\n", num_shards);
    fflush(stdout);
    for (int shard_id = 0; shard_id < num_shards; shard_id++) {
        const auto& global_ids = partition_to_global[shard_id];
        if (global_ids.empty()) {
            if (verbose) {
                printf("  Shard %d: empty, skipping\n", shard_id);
                fflush(stdout);
            }
            continue;
        }

        printf("  [Building HNSW] Shard %d/%d: building index for %zd vectors...\n",
               shard_id + 1, num_shards, global_ids.size());
        fflush(stdout);

        // Create shard index
        shard_indices[shard_id] = new IndexHNSWFlat(d, hnsw_M, metric_type);
        static_cast<IndexHNSWFlat*>(shard_indices[shard_id])->hnsw.efConstruction = ef_construction;
        static_cast<IndexHNSWFlat*>(shard_indices[shard_id])->hnsw.efSearch = ef_search;
        shard_indices[shard_id]->verbose = false; // reduce noise

        // Collect shard vectors
        std::vector<float> shard_vectors(global_ids.size() * d);
        for (size_t i = 0; i < global_ids.size(); i++) {
            idx_t global_id = global_ids[i];
            std::copy(x + global_id * d, x + (global_id + 1) * d,
                     shard_vectors.data() + i * d);
        }

        // Add vectors to shard index
        auto start_time = std::chrono::high_resolution_clock::now();
        shard_indices[shard_id]->add(global_ids.size(), shard_vectors.data());
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        printf("  [Building HNSW] Shard %d/%d: completed in %ld seconds\n",
               shard_id + 1, num_shards, duration);
        fflush(stdout);
    }
    printf("  [Building HNSW] All %d shard indices built successfully\n", num_shards);
    fflush(stdout);
}

void IndexShardBase::build_shard_indices_with_boundaries(
    const float* x, idx_t n,
    const std::vector<int>& assignments,
    const std::vector<idx_t>& boundary_vectors) {

    if (verbose) {
        printf("IndexShard: building %d shard indices with boundary vectors\n", num_shards);
    }

    // Initialize shard structures
    for (auto* idx : shard_indices) {
        delete idx;
    }
    shard_indices.clear();
    shard_indices.resize(num_shards, nullptr);
    partition_to_global.resize(num_shards);
    vector_to_shard = assignments;

    // Group vectors by shard with progress tracking
    printf("  [Grouping Vectors] Starting assignment to %d shards...\n", num_shards);
    fflush(stdout);
    constexpr idx_t PROGRESS_INTERVAL = 100000;
    idx_t next_progress = PROGRESS_INTERVAL;

    for (idx_t i = 0; i < n; i++) {
        int shard_id = assignments[i];
        if (shard_id >= 0 && shard_id < num_shards) {
            partition_to_global[shard_id].push_back(i);
        }

        // Progress reporting
        if (i >= next_progress) {
            printf("  [Grouping Vectors] Processed %zd / %zd vectors (%.1f%%)\n",
                   i, n, 100.0 * i / n);
            fflush(stdout);
            next_progress += PROGRESS_INTERVAL;
        }
    }
    printf("  [Grouping Vectors] Complete: %zd vectors assigned\n", n);
    fflush(stdout);

    // Add boundary vectors to their secondary shards BEFORE building HNSW
    // (runtime toggle: only when boundary replication is enabled)
    if (partition_nested && !boundary_vectors.empty()) {
        printf("  [Adding Boundary Vectors] Adding %zu boundary vectors to partitions before HNSW build...\n",
               boundary_vectors.size());
        fflush(stdout);

        // Find secondary shard for each boundary vector
        IndexFlatL2 centroid_index(d);
        centroid_index.add(num_shards, shard_centroids.data());

        std::vector<int> secondary_counts(num_shards, 0);
        next_progress = PROGRESS_INTERVAL;

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

            // Add to secondary shard's partition
            if (secondary_shard >= 0 && secondary_shard < num_shards &&
                secondary_shard != primary_shard) {
                partition_to_global[secondary_shard].push_back(vec_id);
                secondary_counts[secondary_shard]++;
            }

            // Progress reporting
            if (i > 0 && i >= next_progress) {
                printf("  [Adding Boundary Vectors] Processed %zu / %zu vectors (%.1f%%)\n",
                       i, boundary_vectors.size(), 100.0 * i / boundary_vectors.size());
                fflush(stdout);
                next_progress += PROGRESS_INTERVAL;
            }
        }

        printf("  [Adding Boundary Vectors] Summary:");
        for (int i = 0; i < num_shards; i++) {
            if (secondary_counts[i] > 0) {
                printf(" shard_%d=+%d", i, secondary_counts[i]);
            }
        }
        printf("\n");
        fflush(stdout);
    }

    // Build HNSW for each shard (now includes boundary vectors)
    printf("  [Building HNSW] Starting to build %d shard indices...\n", num_shards);
    fflush(stdout);
    for (int shard_id = 0; shard_id < num_shards; shard_id++) {
        const auto& global_ids = partition_to_global[shard_id];
        if (global_ids.empty()) {
            if (verbose) {
                printf("  Shard %d: empty, skipping\n", shard_id);
                fflush(stdout);
            }
            continue;
        }

        printf("  [Building HNSW] Shard %d/%d: building index for %zd vectors...\n",
               shard_id + 1, num_shards, global_ids.size());
        fflush(stdout);

        // Create shard index
        shard_indices[shard_id] = new IndexHNSWFlat(d, hnsw_M, metric_type);
        static_cast<IndexHNSWFlat*>(shard_indices[shard_id])->hnsw.efConstruction = ef_construction;
        static_cast<IndexHNSWFlat*>(shard_indices[shard_id])->hnsw.efSearch = ef_search;
        shard_indices[shard_id]->verbose = false; // reduce noise

        // Collect shard vectors (including boundary vectors)
        std::vector<float> shard_vectors(global_ids.size() * d);
        for (size_t i = 0; i < global_ids.size(); i++) {
            idx_t global_id = global_ids[i];
            std::copy(x + global_id * d, x + (global_id + 1) * d,
                     shard_vectors.data() + i * d);
        }

        // Add vectors to shard index
        auto start_time = std::chrono::high_resolution_clock::now();
        shard_indices[shard_id]->add(global_ids.size(), shard_vectors.data());
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        printf("  [Building HNSW] Shard %d/%d: completed in %ld seconds\n",
               shard_id + 1, num_shards, duration);
        fflush(stdout);
    }
    printf("  [Building HNSW] All %d shard indices built successfully\n", num_shards);
    fflush(stdout);
}

void IndexShardBase::compute_shard_centroids(const float* x, idx_t n,
                                            const std::vector<int>& assignments) {

    // KH: Debug - print num_samples_per_shard value
    printf("[DEBUG] compute_shard_centroids: num_samples_per_shard = %d\n", num_samples_per_shard);

#ifdef SAMPLING_KMEANS
    // =========================================================================
    // KH: Sub-clustering mode - use k-means within each shard to generate
    // representative centroids for the meta index
    // =========================================================================
    if (num_samples_per_shard > 0) {
        if (verbose) {
            printf("IndexShard: using SUB-CLUSTERING mode (%d sub-clusters per shard, %d iterations)\n",
                   SUB_CLUSTERS_PER_SHARD, SUB_CLUSTERING_NITER);
        }

        // 1. Collect vectors for each shard
        std::vector<std::vector<idx_t>> shard_vector_ids(num_shards);
        for (idx_t i = 0; i < n; i++) {
            int shard_id = assignments[i];
            if (shard_id >= 0 && shard_id < num_shards) {
                shard_vector_ids[shard_id].push_back(i);
            }
        }

        // 2. Perform sub-clustering within each shard
        shard_centroids.clear();
        meta_id_to_shard.clear();

        printf("  [Sub-clustering] Starting sub-clustering for %d shards...\n", num_shards);
        for (int shard_id = 0; shard_id < num_shards; shard_id++) {
            const auto& vec_ids = shard_vector_ids[shard_id];

            if (vec_ids.empty()) {
                // Empty shard: fill with zero vectors
                printf("  [Sub-clustering] Shard %d/%d: empty, filling with zeros\n",
                       shard_id + 1, num_shards);
                for (int s = 0; s < SUB_CLUSTERS_PER_SHARD; s++) {
                    shard_centroids.insert(shard_centroids.end(), d, 0.0f);
                    meta_id_to_shard.push_back(shard_id);
                }
                continue;
            }

            printf("  [Sub-clustering] Shard %d/%d: processing %zu vectors -> %d sub-clusters...\n",
                   shard_id + 1, num_shards, vec_ids.size(),
                   std::min(SUB_CLUSTERS_PER_SHARD, (int)vec_ids.size()));

            // Collect shard vectors into contiguous array
            std::vector<float> shard_vectors(vec_ids.size() * d);
            for (size_t i = 0; i < vec_ids.size(); i++) {
                idx_t vec_idx = vec_ids[i];
                std::copy(x + vec_idx * d, x + (vec_idx + 1) * d,
                         shard_vectors.data() + i * d);
            }

            // Determine actual number of sub-clusters (can't exceed shard size)
            int actual_sub_clusters = std::min(SUB_CLUSTERS_PER_SHARD, (int)vec_ids.size());

            // Perform k-means sub-clustering
            ClusteringParameters cp;
            cp.niter = SUB_CLUSTERING_NITER;
            cp.verbose = false;  // Reduce noise
            cp.seed = 42 + shard_id;  // Different seed per shard for variety

            Clustering sub_clustering(d, actual_sub_clusters, cp);
            IndexFlatL2 sub_index(d);

            auto start_time = std::chrono::high_resolution_clock::now();
            sub_clustering.train(vec_ids.size(), shard_vectors.data(), sub_index);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

            printf("  [Sub-clustering] Shard %d/%d: completed in %ld seconds\n",
                   shard_id + 1, num_shards, duration);

            // Add sub-cluster centroids to meta index vectors
            shard_centroids.insert(shard_centroids.end(),
                                   sub_clustering.centroids.begin(),
                                   sub_clustering.centroids.end());

            // Map each sub-cluster centroid to this shard
            for (int s = 0; s < actual_sub_clusters; s++) {
                meta_id_to_shard.push_back(shard_id);
            }

            // Replicate last centroid if shard has fewer vectors than requested
            if (actual_sub_clusters < SUB_CLUSTERS_PER_SHARD) {
                const float* last_centroid = sub_clustering.centroids.data() +
                                             (actual_sub_clusters - 1) * d;
                for (int s = actual_sub_clusters; s < SUB_CLUSTERS_PER_SHARD; s++) {
                    shard_centroids.insert(shard_centroids.end(),
                                          last_centroid, last_centroid + d);
                    meta_id_to_shard.push_back(shard_id);
                }
            }
        }

        if (verbose) {
            printf("IndexShard: sub-clustering generated %zu vectors for meta index\n",
                   meta_id_to_shard.size());
        }
        return;
    }

#else  // SAMPLING_KMEANS not defined - use random sampling

    // =========================================================================
    // KH: Random sampling mode - sample random points from each shard
    // =========================================================================
    if (num_samples_per_shard > 0) {
        if (verbose) {
            printf("IndexShard: using RANDOM SAMPLING mode (%d samples per shard)\n",
                   num_samples_per_shard);
        }

        // 1. Collect vector indices for each shard
        std::vector<std::vector<idx_t>> shard_vector_ids(num_shards);
        for (idx_t i = 0; i < n; i++) {
            int shard_id = assignments[i];
            if (shard_id >= 0 && shard_id < num_shards) {
                shard_vector_ids[shard_id].push_back(i);
            }
        }

        // 2. Sample random vectors from each shard
        shard_centroids.clear();
        meta_id_to_shard.clear();

        std::mt19937 rng(142);  // Fixed seed for reproducibility

        for (int shard_id = 0; shard_id < num_shards; shard_id++) {
            const auto& vec_ids = shard_vector_ids[shard_id];

            if (vec_ids.empty()) {
                // Empty shard: fill with zero vectors
                if (verbose) {
                    printf("  Warning: Shard %d is empty, filling with zeros\n", shard_id);
                }
                for (int s = 0; s < num_samples_per_shard; s++) {
                    shard_centroids.insert(shard_centroids.end(), d, 0.0f);
                    meta_id_to_shard.push_back(shard_id);
                }
                continue;
            }

            // Actual number of samples (may be less than requested if shard is small)
            int actual_samples = std::min(num_samples_per_shard, (int)vec_ids.size());

            // Fisher-Yates shuffle to randomly select first actual_samples
            std::vector<idx_t> shuffled = vec_ids;
            for (int i = 0; i < actual_samples; i++) {
                std::uniform_int_distribution<size_t> dist(i, shuffled.size() - 1);
                size_t j = dist(rng);
                std::swap(shuffled[i], shuffled[j]);
            }

            // Add sampled vectors to shard_centroids
            for (int s = 0; s < actual_samples; s++) {
                idx_t vec_idx = shuffled[s];
                const float* vec = x + vec_idx * d;
                shard_centroids.insert(shard_centroids.end(), vec, vec + d);
                meta_id_to_shard.push_back(shard_id);
            }

            // Replicate last vector if shard has fewer vectors than requested
            if (actual_samples < num_samples_per_shard) {
                const float* last_vec = x + shuffled[actual_samples - 1] * d;
                for (int s = actual_samples; s < num_samples_per_shard; s++) {
                    shard_centroids.insert(shard_centroids.end(), last_vec, last_vec + d);
                    meta_id_to_shard.push_back(shard_id);
                }
            }
        }

        if (verbose) {
            printf("IndexShard: sampled total %zu vectors for meta index\n",
                   meta_id_to_shard.size());
        }
        return;
    }

#endif  // SAMPLING_KMEANS

    // Original centroid mode (when num_samples_per_shard == 0)
    if (verbose) {
        printf("IndexShard: computing shard centroids (centroid mode)\n");
    }

    shard_centroids.assign(num_shards * d, 0.0f);
    std::vector<int> counts(num_shards, 0);

    // Sum vectors by shard
    for (idx_t i = 0; i < n; i++) {
        int shard_id = assignments[i];
        if (shard_id >= 0 && shard_id < num_shards) {
            for (int j = 0; j < d; j++) {
                shard_centroids[shard_id * d + j] += x[i * d + j];
            }
            counts[shard_id]++;
        }
    }

    // Compute averages
    for (int shard_id = 0; shard_id < num_shards; shard_id++) {
        if (counts[shard_id] > 0) {
            float inv_count = 1.0f / counts[shard_id];
            for (int j = 0; j < d; j++) {
                shard_centroids[shard_id * d + j] *= inv_count;
            }
        }
    }
}

std::vector<int> IndexShardBase::route_query(const float* query) const {
    FAISS_THROW_IF_NOT_MSG(meta_index, "Meta index not built");

    // KH: Determine search size based on mode (sampling vs centroid)
    size_t search_k = (num_samples_per_shard > 0) ?
                      meta_id_to_shard.size() :
                      num_shards;

    std::vector<float> meta_distances(search_k);
    std::vector<idx_t> meta_labels(search_k);

    // Search meta index
    meta_index->search(1, query, search_k, meta_distances.data(), meta_labels.data());

    std::vector<int> selected_shards;

    // KH: Centroid mode - meta label is directly shard ID
    if (num_samples_per_shard == 0) {
        for (size_t i = 0; i < search_k; i++) {
            idx_t shard_id = meta_labels[i];
            if (shard_id >= 0 && shard_id < num_shards) {
                selected_shards.push_back(static_cast<int>(shard_id));
            }
        }
    }
    // KH: Sampling mode - map meta label to shard ID with deduplication
    else {
        std::set<int> unique_shards;
        for (size_t i = 0; i < search_k && unique_shards.size() < (size_t)num_shards; i++) {
            idx_t meta_id = meta_labels[i];
            if (meta_id >= 0 && meta_id < meta_id_to_shard.size()) {
                int shard_id = meta_id_to_shard[meta_id];
                unique_shards.insert(shard_id);
            }
        }
        selected_shards.assign(unique_shards.begin(), unique_shards.end());
    }

    return selected_shards;
}

void IndexShardBase::search_shard(
        int shard_id,
        const float* query,
        int k,
        std::vector<idx_t>& indices,
        std::vector<float>& distances) const {
    
    indices.clear();
    distances.clear();
    
#ifdef FAISS_WITH_DOCA
    // DPU shard (shard_id == 1) and DOCA initialized
    if (shard_id == 1 && doca_initialized) {
        // Send search request to DPU
        std::vector<uint32_t> dpu_ids = {static_cast<uint32_t>(shard_id)};
        doca_error_t ret = send_search_to_dpu(query, dpu_ids, k);
        
        if (ret == DOCA_SUCCESS) {
            // Receive results from DPU
            auto results = receive_search_from_dpu();
            
            // DPU already returns global IDs, use them directly
            for (const auto& rp : results) {
                indices.push_back(static_cast<idx_t>(rp.id));
                distances.push_back(rp.dist);
            }
            return;
        }
        // If DMA fails, fall through to host search
        DOCA_LOG_WARN("DPU search failed, falling back to host search");
    }
#endif
    
    if (shard_id < 0 || shard_id >= num_shards || !shard_indices[shard_id]) {
        return;
    }
    
    const auto& global_ids = partition_to_global[shard_id];
    int shard_size = static_cast<int>(global_ids.size());
    int local_k = std::min(k, shard_size);
    
    if (local_k == 0) {
        return;
    }
    
    // Search within shard
    std::vector<float> local_distances(local_k);
    std::vector<idx_t> local_indices(local_k);
    
    shard_indices[shard_id]->search(1, query, local_k, 
                                   local_distances.data(), local_indices.data());
    
    // Convert local IDs to global IDs
    for (int i = 0; i < local_k; i++) {
        idx_t local_id = local_indices[i];
        if (local_id >= 0 && local_id < shard_size) {
            indices.push_back(global_ids[local_id]);
            distances.push_back(local_distances[i]);
        }
    }
}

void IndexShardBase::add_to_existing_shards(idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index not trained");
    FAISS_THROW_IF_NOT_MSG(meta_index, "Meta index not built");
    
    // Route each vector to nearest shard
    std::vector<std::vector<float>> shard_vectors(num_shards);
    std::vector<std::vector<idx_t>> shard_new_global_ids(num_shards);
    
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        
        // Find nearest shard using meta index
        std::vector<float> meta_dist(1);
        std::vector<idx_t> meta_label(1);
        meta_index->search(1, vec, 1, meta_dist.data(), meta_label.data());
        
        int shard_id = static_cast<int>(meta_label[0]);
        if (shard_id >= 0 && shard_id < num_shards) {
            // Add to shard's vector list
            shard_vectors[shard_id].insert(shard_vectors[shard_id].end(), vec, vec + d);
            
            // Assign global ID
            idx_t global_id = (xids) ? xids[i] : (ntotal + i);
            shard_new_global_ids[shard_id].push_back(global_id);
        }
    }
    
    // Add vectors to each shard
    for (int shard_id = 0; shard_id < num_shards; shard_id++) {
        if (shard_vectors[shard_id].empty()) continue;
        
        size_t shard_n = shard_vectors[shard_id].size() / d;
        if (verbose) {
            printf("  Adding %zd vectors to shard %d\n", shard_n, shard_id);
        }
        
        // Create shard index if it doesn't exist
        if (!shard_indices[shard_id]) {
            shard_indices[shard_id] = new IndexHNSWFlat(d, hnsw_M, metric_type);
            static_cast<IndexHNSWFlat*>(shard_indices[shard_id])->hnsw.efConstruction = ef_construction;
            static_cast<IndexHNSWFlat*>(shard_indices[shard_id])->hnsw.efSearch = ef_search;
            shard_indices[shard_id]->verbose = false;
        }
        
        // Add vectors to shard
        shard_indices[shard_id]->add(shard_n, shard_vectors[shard_id].data());
        
        // Update global ID mapping
        for (idx_t global_id : shard_new_global_ids[shard_id]) {
            partition_to_global[shard_id].push_back(global_id);
            vector_to_shard.push_back(shard_id);
        }
    }
    
    ntotal += n;
}

void IndexShardBase::merge_shard_results(
        const std::vector<std::vector<idx_t>>& shard_indices,
        const std::vector<std::vector<float>>& shard_distances,
        idx_t k,
        idx_t* labels,
        float* distances) const {
    
    // Collect all candidates
    std::vector<std::pair<float, idx_t>> candidates;
    for (size_t i = 0; i < shard_indices.size(); i++) {
        for (size_t j = 0; j < shard_indices[i].size(); j++) {
            candidates.emplace_back(shard_distances[i][j], shard_indices[i][j]);
        }
    }
    
    if (candidates.empty()) {
        // Fill with -1 if no results
        for (int i = 0; i < k; i++) {
            labels[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
        }
        return;
    }
    
    // Sort by distance
    std::sort(candidates.begin(), candidates.end());
    
    // Take top-k
    int result_count = std::min(static_cast<int>(candidates.size()), static_cast<int>(k));
    
    for (int i = 0; i < result_count; i++) {
        distances[i] = candidates[i].first;
        labels[i] = candidates[i].second;
    }
    
    // Fill remaining with -1
    for (int i = result_count; i < k; i++) {
        distances[i] = std::numeric_limits<float>::max();
        labels[i] = -1;
    }
}

void IndexShardBase::search_batch(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int effective_nprobe) const {

    // Timing measurements
    auto t_total_start = std::chrono::high_resolution_clock::now();

    // Route all queries to get shard selections
    std::vector<std::vector<int>> selected_shards_2d(n);

    // Meta search for all queries
    auto t_routing_start = std::chrono::high_resolution_clock::now();
    std::vector<float> meta_distances(n * effective_nprobe);
    std::vector<idx_t> meta_labels(n * effective_nprobe);

    meta_index->search(n, x, effective_nprobe, meta_distances.data(), meta_labels.data());
    auto t_routing_end = std::chrono::high_resolution_clock::now();
    
    // Parse meta results
    auto t_parse_start = std::chrono::high_resolution_clock::now();
    for (int qi = 0; qi < n; ++qi) {
        const idx_t* ids = meta_labels.data() + (size_t)qi * effective_nprobe;
        auto& sel = selected_shards_2d[qi];
        sel.reserve(effective_nprobe);

        for (int j = 0; j < effective_nprobe; ++j) {
            idx_t meta_id = ids[j];
            int shard_id;

            // KH: Convert meta_id to shard_id (sampling mode support)
            if (num_samples_per_shard > 0 && !meta_id_to_shard.empty()) {
                // Sampling mode: use mapping
                if (meta_id >= 0 && meta_id < (idx_t)meta_id_to_shard.size()) {
                    shard_id = meta_id_to_shard[meta_id];
                } else {
                    continue;  // Invalid meta_id
                }
            } else {
                // Centroid mode: meta_id == shard_id
                shard_id = (int)meta_id;
            }

            if (shard_id >= 0 && shard_id < num_shards) {
                sel.push_back(shard_id);
#ifdef FAISS_WITH_DOCA
                // Log query-to-shard mapping
                //DOCA_LOG_INFO("[QUERY-ROUTING] Query %d -> Shard %d (meta_id=%ld, meta_dist=%f)",qi, shard_id, meta_id, meta_distances[qi * effective_nprobe + j]);
#endif
            }
        }
    }

    // Separate host and DPU shards
    std::vector<std::vector<int>> host_ids_2d(n);
    std::vector<std::vector<uint32_t>> dpu_ids_2d(n);

    bool need_dpu = false;
    bool need_host = false;
    int num_host_queries = 0;
    int num_dpu_queries = 0;
    
    for (int qi = 0; qi < n; ++qi) {
        const auto& sel = selected_shards_2d[qi];
        host_ids_2d[qi].reserve(sel.size());
        dpu_ids_2d[qi].reserve(sel.size());

        bool query_uses_host = false;
        bool query_uses_dpu = false;

        for (int sid : sel) {
            if (sid < 0 || sid >= num_shards) continue;
#ifdef FAISS_WITH_DOCA
            // Shard 1 goes to DPU (matches DPU dpu_2_static_ids = {1})
            if (sid == 1 && doca_initialized) {
                dpu_ids_2d[qi].push_back((uint32_t)sid);
                need_dpu = true;
                query_uses_dpu = true;
            } else {
#endif
                host_ids_2d[qi].push_back(sid);
                need_host = true;
                query_uses_host = true;
#ifdef FAISS_WITH_DOCA
            }
#endif
        }

        if (query_uses_host) num_host_queries++;
        if (query_uses_dpu) num_dpu_queries++;
    }
    auto t_parse_end = std::chrono::high_resolution_clock::now();
    
    // Collect results per query
    std::vector<std::vector<idx_t>> per_query_indices(n);
    std::vector<std::vector<float>> per_query_distances(n);

#ifdef FAISS_WITH_DOCA
    // === Async DPU send ===
    auto t_dpu_start = std::chrono::high_resolution_clock::now();
    std::thread dpu_send_thread;
    bool dpu_started = false;
    DPUSendResult dpu_send_result = {DOCA_SUCCESS, 0.0};
    std::unordered_map<uint32_t, std::vector<int>> dpu_shard_to_queries;

    if (need_dpu) {
        // Group queries by shard
        for (int qi = 0; qi < n; ++qi) {
            for (uint32_t sid : dpu_ids_2d[qi]) {
                dpu_shard_to_queries[sid].push_back(qi);
            }
        }

        if (!dpu_shard_to_queries.empty()) {
            // Start async DPU send in background thread
            dpu_send_thread = std::thread([&]() {
                dpu_send_result = send_batch_search_to_dpu(x, dpu_shard_to_queries, k);
            });
            dpu_started = true;
        }
    }
#endif

    // Handle host shards (runs in parallel with DPU send!)
    auto t_host_start = std::chrono::high_resolution_clock::now();
    if (need_host) {
        std::unordered_map<int, std::vector<int>> host_shard_to_queries;

        // Group queries by shard
        for (int qi = 0; qi < n; ++qi) {
            for (int sid : host_ids_2d[qi]) {
                host_shard_to_queries[sid].push_back(qi);
            }
        }

        // Process each host shard
        for (const auto& [shard_id, query_list] : host_shard_to_queries) {
            if (shard_id < 0 || shard_id >= num_shards || !shard_indices[shard_id]) {
                continue;
            }

            const auto& global_ids = partition_to_global[shard_id];
            int shard_size = static_cast<int>(global_ids.size());
            int local_k = std::min(k, (idx_t)shard_size);

            if (local_k == 0) continue;

            // Collect queries for this shard
            std::vector<float> shard_queries;
            shard_queries.reserve(query_list.size() * d);

            for (int qi : query_list) {
                const float* query = x + qi * d;
                shard_queries.insert(shard_queries.end(), query, query + d);
            }

            // Batch search on shard with timing
            auto t_shard_start = std::chrono::high_resolution_clock::now();
            std::vector<float> local_distances(query_list.size() * local_k);
            std::vector<idx_t> local_indices(query_list.size() * local_k);

            shard_indices[shard_id]->search(query_list.size(), shard_queries.data(),
                                           local_k, local_distances.data(), local_indices.data());
            auto t_shard_end = std::chrono::high_resolution_clock::now();
            double shard_ms = std::chrono::duration<double, std::milli>(t_shard_end - t_shard_start).count();

            printf("[SHARD %d] Processed %zu queries in %.3f ms (%.3f ms/query, shard_size=%d)\n",
                   shard_id, query_list.size(), shard_ms, shard_ms / query_list.size(), shard_size);

            // Convert local IDs to global IDs and distribute to queries
            for (size_t i = 0; i < query_list.size(); ++i) {
                int qi = query_list[i];
                auto& indices = per_query_indices[qi];
                auto& dists = per_query_distances[qi];

                for (int j = 0; j < local_k; ++j) {
                    idx_t local_id = local_indices[i * local_k + j];
                    if (local_id >= 0 && local_id < shard_size) {
                        indices.push_back(global_ids[local_id]);
                        dists.push_back(local_distances[i * local_k + j]);
                    }
                }
            }
        }
    }
    auto t_host_end = std::chrono::high_resolution_clock::now();

#ifdef FAISS_WITH_DOCA
    // === Wait for DPU send completion and receive results ===
    // Measure total wait time after host completes
    auto t_wait_for_dpu_start = std::chrono::high_resolution_clock::now();
    auto t_wait_for_dpu_end = t_wait_for_dpu_start;  // Initialize to start time

    if (dpu_started) {
        dpu_send_thread.join();  // Wait for send to complete (may block if DPU send still running)

        if (dpu_send_result.error == DOCA_SUCCESS) {
            // Pass wait_end pointer to get accurate timing (after COMCH negotiation completes)
            auto dpu_results = receive_batch_search_from_dpu(&t_wait_for_dpu_end);

            // Merge DPU results into per_query results
            for (const auto& [qidx, results] : dpu_results) {
                if (qidx < (uint32_t)n) {
                    auto& indices = per_query_indices[qidx];
                    auto& dists = per_query_distances[qidx];

                    // DPU already returns global IDs, use them directly
                    for (const auto& rp : results) {
                        indices.push_back(static_cast<idx_t>(rp.id));
                        dists.push_back(rp.dist);
                    }
                }
            }
        } else {
            DOCA_LOG_ERR("DPU send failed: %s", doca_error_get_descr(dpu_send_result.error));
            // If DPU failed, end time is now
            t_wait_for_dpu_end = std::chrono::high_resolution_clock::now();
        }
    }
#endif

    // Merge final results for each query
    auto t_merge_start = std::chrono::high_resolution_clock::now();
    for (int qi = 0; qi < n; ++qi) {
        std::vector<std::vector<idx_t>> shard_indices = {per_query_indices[qi]};
        std::vector<std::vector<float>> shard_distances = {per_query_distances[qi]};

        merge_shard_results(shard_indices, shard_distances, k,
                           labels + qi * k, distances + qi * k);
    }
    auto t_merge_end = std::chrono::high_resolution_clock::now();
    auto t_total_end = std::chrono::high_resolution_clock::now();

    // === LOG TOP-K RESULTS TO CSV FOR COMPARISON ===
    // Determine mode: host-only or DPU
#ifdef FAISS_WITH_DOCA
    bool is_dpu_mode = (num_dpu_queries > 0);
    const char* mode_str = is_dpu_mode ? "dpu" : "host_only";
#else
    const char* mode_str = "host_only";
#endif

    // CSV logging disabled
    // Log first 10 queries, top-10 results each
    // static int batch_counter = 0;
    // char csv_filename[256];
    // std::snprintf(csv_filename, sizeof(csv_filename),
    //               "/home/dpu/faiss/log/topk_results_%s_batch%d.csv",
    //               mode_str, batch_counter++);

    // std::ofstream csv_file(csv_filename);
    // if (csv_file.is_open()) {
    //     // CSV header
    //     csv_file << "query_id,rank,label,distance\n";

    //     // Log first min(10, n) queries
    //     int queries_to_log = std::min(10, (int)n);
    //     int topk_to_log = std::min(10, (int)k);

    //     for (int qi = 0; qi < queries_to_log; ++qi) {
    //         for (int ki = 0; ki < topk_to_log; ++ki) {
    //             idx_t label = labels[qi * k + ki];
    //             float dist = distances[qi * k + ki];
    //             csv_file << qi << "," << ki << "," << label << "," << dist << "\n";
    //         }
    //     }

    //     csv_file.close();
    //     printf("[TOPK-LOG] Saved top-%d results for %d queries to: %s\n",
    //            topk_to_log, queries_to_log, csv_filename);
    // } else {
    //     fprintf(stderr, "[TOPK-LOG] Failed to open CSV file: %s\n", csv_filename);
    // }

    // Print timing breakdown
    double routing_ms = std::chrono::duration<double, std::milli>(t_routing_end - t_routing_start).count();
    double parse_ms = std::chrono::duration<double, std::milli>(t_parse_end - t_parse_start).count();
    double host_ms = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();
    double merge_ms = std::chrono::duration<double, std::milli>(t_merge_end - t_merge_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

    printf("\n=== Search Batch Timing Breakdown (n=%zd queries, nprobe=%d) ===\n", n, effective_nprobe);
    printf("1. Meta search (routing):           %8.3f ms\n", routing_ms);
    printf("2. Packaging (parse & grouping):   %8.3f ms\n", parse_ms);

#ifdef FAISS_WITH_DOCA
    // Calculate all DPU-related timings
    double dpu_send_actual_ms = dpu_send_result.send_time_ms;  // Actual send time (from inside thread)
    double wait_for_dpu_total_ms = std::chrono::duration<double, std::milli>(t_wait_for_dpu_end - t_wait_for_dpu_start).count();

    if (num_dpu_queries > 0) {
        printf("3. DPU send (async, parallel):      %8.3f ms (actual send time in thread)\n", dpu_send_actual_ms);
    }
#endif

    if (num_host_queries > 0) {
        // Collect host shard IDs for logging
        std::string host_shard_str;
        if (need_host) {
            std::unordered_map<int, std::vector<int>> host_shard_to_queries;
            for (int qi = 0; qi < n; ++qi) {
                for (int sid : host_ids_2d[qi]) {
                    host_shard_to_queries[sid].push_back(qi);
                }
            }

            std::vector<int> host_shards;
            for (const auto& [shard_id, query_list] : host_shard_to_queries) {
                host_shards.push_back(shard_id);
            }
            std::sort(host_shards.begin(), host_shards.end());

            host_shard_str = "[";
            for (size_t i = 0; i < host_shards.size(); ++i) {
                if (i > 0) host_shard_str += ", ";
                host_shard_str += std::to_string(host_shards[i]);
            }
            host_shard_str += "]";
        }

        printf("4. Host search (parallel):          %8.3f ms (%d queries, shards: %s)\n",
               host_ms, num_host_queries, host_shard_str.c_str());
    }

#ifdef FAISS_WITH_DOCA
    if (num_dpu_queries > 0) {
        printf("5. Wait for DPU (total blocking):   %8.3f ms (join + receive + DPU search)\n", wait_for_dpu_total_ms);
    }
#endif

    printf("6. Final merge:                     %8.3f ms\n", merge_ms);
    printf("---\n");
    printf("Total time:                         %8.3f ms\n", total_ms);

#ifdef FAISS_WITH_DOCA
    if (num_dpu_queries > 0) {
        double sum_parts = routing_ms + parse_ms + host_ms + wait_for_dpu_total_ms + merge_ms;
        printf("Sum of parts (1+2+4+5+6):           %8.3f ms (should ≈ total)\n", sum_parts);
        printf("Note: DPU send (3) runs in parallel with Host search (4), not added to sum\n");
    }
#endif

    printf("======================================================\n\n");
}

#ifdef FAISS_WITH_DOCA  // ---- host<->DPU pipeline helpers (require DOCA) ----
//hs_pipelining - 헬퍼 함수 1: Meta routing (쿼리들을 Host/DPU shard로 분류)
void IndexShardBase::do_meta_routing(
        int slot,
        const float* x,
        size_t start_idx,
        size_t batch_n,
        int effective_nprobe) const {

    int batch_idx = start_idx / batch_n;  // batch_idx 계산 (batch_size가 batch_n과 같다고 가정)
    auto t_start = std::chrono::high_resolution_clock::now();

    const float* batch_x = x + start_idx * d;

    // Meta search
    std::vector<float> meta_distances(batch_n * effective_nprobe);
    std::vector<idx_t> meta_labels(batch_n * effective_nprobe);
    meta_index->search(batch_n, batch_x, effective_nprobe, meta_distances.data(), meta_labels.data());

    // Parse meta results → selected_shards_2d
    std::vector<std::vector<int>> selected_shards_2d(batch_n);
    for (size_t qi = 0; qi < batch_n; ++qi) {
        const idx_t* ids = meta_labels.data() + qi * effective_nprobe;
        auto& sel = selected_shards_2d[qi];
        sel.reserve(effective_nprobe);

        for (int j = 0; j < effective_nprobe; ++j) {
            idx_t meta_id = ids[j];
            int shard_id;

            if (num_samples_per_shard > 0 && !meta_id_to_shard.empty()) {
                if (meta_id >= 0 && meta_id < (idx_t)meta_id_to_shard.size()) {
                    shard_id = meta_id_to_shard[meta_id];
                } else {
                    continue;
                }
            } else {
                shard_id = (int)meta_id;
            }

            if (shard_id >= 0 && shard_id < num_shards) {
                sel.push_back(shard_id);
            }
        }
    }

    // Separate host and DPU shards
    batch_host_shard_to_queries[slot].clear();
    batch_dpu_shard_to_queries[slot].clear();

    for (size_t qi = 0; qi < batch_n; ++qi) {
        const auto& sel = selected_shards_2d[qi];
        for (int sid : sel) {
            if (sid < 0 || sid >= num_shards) continue;
#ifdef FAISS_WITH_DOCA
            if (sid == 1 && doca_initialized) {
                batch_dpu_shard_to_queries[slot][(uint32_t)sid].push_back((int)qi);
            } else {
#endif
                batch_host_shard_to_queries[slot][sid].push_back((int)qi);
#ifdef FAISS_WITH_DOCA
            }
#endif
        }
    }

    // Debug: print first 5 DPU query indices
    // printf("[META-ROUTING] slot=%d, batch_n=%zu\n", slot, batch_n);
    for (const auto& kv : batch_dpu_shard_to_queries[slot]) {
        // printf("[META-ROUTING] shard %u: %zu queries, first 5 qidx: ", kv.first, kv.second.size());
        // for (size_t i = 0; i < std::min((size_t)5, kv.second.size()); ++i) {
        //     printf("%d ", kv.second[i]);
        // }
        // printf("\n");
    }

    // 타이밍 및 쿼리 수 저장
    auto t_end = std::chrono::high_resolution_clock::now();
    batch_timings[batch_idx].meta_routing_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Host/DPU 쿼리 수 계산
    int host_q = 0, dpu_q = 0;
    for (const auto& kv : batch_host_shard_to_queries[slot]) {
        host_q += kv.second.size();
    }
    for (const auto& kv : batch_dpu_shard_to_queries[slot]) {
        dpu_q += kv.second.size();
    }
    batch_timings[batch_idx].host_queries = host_q;
    batch_timings[batch_idx].dpu_queries = dpu_q;
}

//hs_pipelining - 헬퍼 함수 2: DPU send (DPU에 배치 요청 전송)
void IndexShardBase::do_dpu_send(int slot, int batch_idx, const float* x, size_t start_idx, idx_t k) const {
    auto t_start = std::chrono::high_resolution_clock::now();

    if (!batch_dpu_shard_to_queries[slot].empty()) {
        //hs_pipelining - DPU로 보낼 쿼리가 있으면 전송
        batch_needs_dpu_recv[batch_idx] = true;  // recv 필요 (batch 기반)
        const float* batch_x = x + start_idx * d;
        send_batch_search_to_dpu(batch_x, batch_dpu_shard_to_queries[slot], k);
        // printf("[DPU_SEND] batch=%d, slot=%d, needs_recv=true\n", batch_idx, slot);
    } else {
        //hs_pipelining - DPU로 보낼 쿼리가 없으면 recv 불필요 표시
        batch_needs_dpu_recv[batch_idx] = false;  // recv 불필요 (batch 기반)
        // printf("[DPU_SEND] batch=%d, slot=%d, needs_recv=false (no DPU queries)\n", batch_idx, slot);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    batch_timings[batch_idx].dpu_send_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

//hs_pipelining - 헬퍼 함수 3: DPU receive (non-blocking polling)
bool IndexShardBase::do_dpu_receive(int slot, int batch_idx) const {
    auto t_start = std::chrono::high_resolution_clock::now();

    //hs_pipelining - 10μs timeout으로 non-blocking polling
    auto timeout = std::chrono::microseconds(10);
    auto dpu_results = receive_batch_search_from_dpu(nullptr, timeout);

    if (dpu_results.empty()) {
        //hs_pipelining - timeout 발생 → 아직 응답 안 옴
        return false;
    }

    //hs_pipelining - 응답 받음 → slot에 저장
    // printf("[DO_DPU_RECV] Storing DPU result to slot=%d, n_queries=%zu\n", slot, dpu_results.size());
    fflush(stdout);
    batch_dpu_results[slot] = std::move(dpu_results);

    auto t_end = std::chrono::high_resolution_clock::now();
    batch_timings[batch_idx].dpu_recv_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    dpu_last_recv_time = t_end;  // Idle time 계산용

    return true;
}

//hs_pipelining - 헬퍼 함수 4: Merge (Host + DPU 결과 병합)
void IndexShardBase::do_merge(
        int slot,
        int batch_idx,
        idx_t k,
        float* distances,
        idx_t* labels,
        size_t start_idx,
        size_t batch_n) const {

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t qi = 0; qi < batch_n; ++qi) {
        size_t global_qi = start_idx + qi;

        //hs_pipelining - Host 결과 가져오기
        auto& host_indices = batch_host_indices[slot][qi];
        auto& host_dists = batch_host_distances[slot][qi];

        //hs_pipelining - DPU 결과 추가
        // printf("[MERGE] qi=%zu, host_indices.size()=%zu, batch_dpu_results[slot].count(qi)=%d\n",
        //        qi, host_indices.size(), (int)batch_dpu_results[slot].count(qi));
        if (batch_dpu_results[slot].count(qi)) {
            // printf("[MERGE]   DPU result found for qi=%zu, n_results=%zu\n",
            //        qi, batch_dpu_results[slot][qi].size());
            for (const auto& rp : batch_dpu_results[slot][qi]) {
                host_indices.push_back(static_cast<idx_t>(rp.id));
                host_dists.push_back(rp.dist);
            }
        }

        //hs_pipelining - merge_shard_results로 top-k 선정
        std::vector<std::vector<idx_t>> shard_indices = {host_indices};
        std::vector<std::vector<float>> shard_distances = {host_dists};
        merge_shard_results(shard_indices, shard_distances, k,
                           labels + global_qi * k,
                           distances + global_qi * k);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    batch_timings[batch_idx].merge_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // printf("[MERGE] === Batch merge COMPLETED: slot=%d, start_idx=%zu, batch_n=%zu ===\n",
    //        slot, start_idx, batch_n);
    fflush(stdout);

    //hs_pipelining - 슬롯 플래그 리셋은 send 전에 수행 (여기서 하면 다음 배치 플래그 충돌)
}

//hs_pipelining - 헬퍼 함수 5: Schedule thread 메인 루프
void IndexShardBase::schedule_thread_func(
        int total_batches,
        idx_t n,
        const float* x,
        int batch_size,
        idx_t k,
        int effective_nprobe,
        float* distances,
        idx_t* labels,
        std::queue<size_t>& batch_queue) const {

    int schedule_batch_idx = 0;  // 다음에 send할 배치 번호
    int recv_batch_idx = 0;      // 다음에 receive할 배치 번호 (DPU는 순차 응답)
    int merge_batch_idx = 0;     // 다음에 merge할 배치 번호

    //hs_pipelining - 첫 배치는 조건 없이 바로 시작
    if (!batch_queue.empty()) {
        size_t start_idx = batch_queue.front();
        batch_queue.pop();
        size_t batch_n = std::min((size_t)batch_size, n - start_idx);
        int slot = 0;

        //hs_pipelining - send 전 슬롯 초기화 (이전 배치 데이터/플래그 충돌 방지)
        batch_host_done[slot].store(false);
        batch_dpu_results[slot].clear();  // 이전 DPU 결과 제거

        do_meta_routing(slot, x, start_idx, batch_n, effective_nprobe);
        do_dpu_send(slot, schedule_batch_idx, x, start_idx, k);  // batch_idx 전달
        host_can_start[slot].store(true);

        schedule_batch_idx++;
    }

    //hs_pipelining - 이후 배치들은 루프에서 조건부 처리
    while (merge_batch_idx < total_batches) {
        int merge_slot = merge_batch_idx % MAX_INFLIGHT_BATCHES;

        //hs_pipelining - 1. DPU receive polling (non-blocking, 순차적으로 receive)
        // recv_batch_idx < schedule_batch_idx: send한 배치만 receive 시도
        // store_slot: batch 기반 (merge에서 batch_dpu_results[merge_slot]로 읽으므로 일치해야 함)
        int store_slot = recv_batch_idx % MAX_INFLIGHT_BATCHES;
        if (recv_batch_idx < schedule_batch_idx) {
            if (!batch_needs_dpu_recv[recv_batch_idx]) {
                // DPU 쿼리 없는 배치 → recv 불필요, 바로 다음 배치로
                // printf("[RECV-SKIP] batch=%d has no DPU queries, skipping recv\n", recv_batch_idx);
                recv_batch_idx++;
            } else {
                // DPU 쿼리 있는 배치 → recv 시도
                // DPU는 순서대로 보내므로 recv 자체는 순서대로 받음 (dma_pool.recv_slot 내부에서 관리)
                // 저장은 batch 기반 slot에 해서 merge와 일치시킴
                if (do_dpu_receive(store_slot, recv_batch_idx)) {
                    // printf("[RECV-OK] batch=%d recv completed\n", recv_batch_idx);
                    recv_batch_idx++;  // 성공 시 다음 배치로
                }
            }
        }

        //hs_pipelining - 2. 다음 배치 send 조건 체크
        if (schedule_batch_idx < total_batches && !batch_queue.empty()) {
            int slot = schedule_batch_idx % MAX_INFLIGHT_BATCHES;

            // slot 재사용 안전 조건: 이 slot을 사용하던 이전 배치가 merge 완료됐는지 확인
            // 예: batch 4가 slot 0 사용하려면, batch 0이 merge 완료되어야 함
            int prev_batch_using_slot = schedule_batch_idx - MAX_INFLIGHT_BATCHES;
            bool slot_free = (prev_batch_using_slot < 0) || (prev_batch_using_slot < merge_batch_idx);

            if (slot_free) {
                size_t start_idx = batch_queue.front();
                batch_queue.pop();
                size_t batch_n = std::min((size_t)batch_size, n - start_idx);

                //hs_pipelining - send 전 슬롯 초기화 (이전 배치 데이터/플래그 충돌 방지)
                batch_host_done[slot].store(false);
                batch_dpu_results[slot].clear();  // 이전 DPU 결과 제거

                do_meta_routing(slot, x, start_idx, batch_n, effective_nprobe);
                do_dpu_send(slot, schedule_batch_idx, x, start_idx, k);  // batch_idx 전달

                // Collision 체크: send 중에 DPU result를 이미 받았으면 파싱 후 recv_batch_idx 증가
                if (dma_cfg.collision_recv_done) {
                    int store_slot = recv_batch_idx % MAX_INFLIGHT_BATCHES;
                    // printf("[COLLISION] batch=%d: recv during send, storing to slot=%d, recv_batch_idx++ (%d→%d)\n",
                    //        schedule_batch_idx, store_slot, recv_batch_idx, recv_batch_idx + 1);

                    // Parse DPU result from pending_recv_buffer
                    const uint8_t* rbuf = reinterpret_cast<const uint8_t*>(dma_cfg.pending_recv_buffer);
                    size_t total_bytes_resp = dma_cfg.file_size;
                    const uint8_t* p = rbuf;

                    if (total_bytes_resp >= sizeof(uint32_t)) {
                        uint32_t n_blocks = *reinterpret_cast<const uint32_t*>(p);
                        p += sizeof(uint32_t);

                        // printf("[COLLISION-PARSE] n_blocks=%u, total_bytes=%zu, store_slot=%d\n",
                        //        n_blocks, total_bytes_resp, store_slot);
                        // printf("[COLLISION-PARSE] DPU returned qidx: ");

                        auto& results = batch_dpu_results[store_slot];

                        for (uint32_t b = 0; b < n_blocks; ++b) {
                            if (p + 2 * sizeof(uint32_t) > rbuf + total_bytes_resp) {
                                // printf("\n[COLLISION] Invalid response format at block %u\n", b);
                                break;
                            }

                            uint32_t qidx = *reinterpret_cast<const uint32_t*>(p);
                            p += sizeof(uint32_t);

                            uint32_t n_i = *reinterpret_cast<const uint32_t*>(p);
                            p += sizeof(uint32_t);

                            // printf("%u ", qidx);

                            if (p + n_i * sizeof(ResultPair) > rbuf + total_bytes_resp) {
                                // printf("\n[COLLISION] Invalid response format at block %u\n", b);
                                break;
                            }

                            const ResultPair* rp = reinterpret_cast<const ResultPair*>(p);
                            auto& vec = results[qidx];
                            vec.reserve(vec.size() + n_i);

                            for (uint32_t i = 0; i < n_i; ++i) {
                                vec.push_back({rp[i].dist, rp[i].id});
                            }
                            p += n_i * sizeof(ResultPair);
                        }
                        // printf("\n");
                        fflush(stdout);

                        // printf("[COLLISION] Stored %u blocks to batch_dpu_results[%d]\n", n_blocks, store_slot);
                    } else {
                        // printf("[COLLISION] Response too small: file_size=%zu\n", total_bytes_resp);
                    }

                    recv_batch_idx++;
                    const_cast<dma_copy_cfg&>(dma_cfg).collision_recv_done = false;
                }

                host_can_start[slot].store(true);

                schedule_batch_idx++;
            }
        }

        //hs_pipelining - 3. Merge 조건 체크
        //               핵심: merge_batch_idx < recv_batch_idx 여야만 해당 batch의 DPU recv가 완료된 것
        //               DPU 쿼리 없는 배치도 recv polling에서 recv_batch_idx 증가시킴
        bool host_done = batch_host_done[merge_slot].load();
        bool dpu_done = (merge_batch_idx < recv_batch_idx);  // batch 기반 (slot 기반 아님!)

        // printf("[SCHED] merge_batch=%d, merge_slot=%d, host_done=%d, recv_batch_idx=%d, dpu_done=%d\n",
        //        merge_batch_idx, merge_slot, host_done, recv_batch_idx, dpu_done);
        // fflush(stdout);

        if (host_done && dpu_done) {
            size_t start_idx = merge_batch_idx * batch_size;
            size_t batch_n = std::min((size_t)batch_size, n - start_idx);
            do_merge(merge_slot, merge_batch_idx, k, distances, labels, start_idx, batch_n);
            merge_batch_idx++;
            continue;
        }

        //hs_pipelining - busy-wait 방지 (CPU 양보)
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}
#endif  // FAISS_WITH_DOCA (pipeline helpers)

//hs_pipelining - 파이프라인 검색 함수
void IndexShardBase::search_pipelined(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int batch_size,
        std::queue<size_t>& batch_queue) const {

#ifdef FAISS_WITH_DOCA
    auto t_total_start = std::chrono::high_resolution_clock::now();

    int total_batches = batch_queue.size();
    int effective_nprobe = nprobe > 0 ? nprobe : 1;

    printf("\n=== search_pipelined START (n=%zd, batch_size=%d, total_batches=%d) ===\n",
           n, batch_size, total_batches);

    //hs_pipelining - 플래그 초기화
    for (int i = 0; i < MAX_INFLIGHT_BATCHES; i++) {
        batch_host_done[i].store(false);
        host_can_start[i].store(false);
    }
    //hs_pipelining - batch 기반 플래그 및 타이밍 초기화
    for (int i = 0; i < MAX_TOTAL_BATCHES; i++) {
        batch_needs_dpu_recv[i] = false;
        batch_timings[i] = BatchTiming{};  // 타이밍 초기화
    }

    //hs_pipelining - slot 카운터는 리셋하지 않음
    //               warmup에서 이미 slot을 사용했으므로, DPU와 동기화를 유지하려면
    //               현재 값을 그대로 사용해야 함
    printf("[PIPELINING] Slot counters NOT reset: send_slot=%d, recv_slot=%d (continuing from warmup)\n",
           dma_pool.send_slot, dma_pool.recv_slot);

    //hs_pipelining - Schedule thread 시작
    std::thread schedule_thread(&IndexShardBase::schedule_thread_func, this,
                                total_batches, n, x, batch_size, k, effective_nprobe,
                                distances, labels, std::ref(batch_queue));

    //hs_pipelining - Main thread: Host search 루프
    int host_batch_idx = 0;
    while (host_batch_idx < total_batches) {
        int slot = host_batch_idx % MAX_INFLIGHT_BATCHES;

        // host_can_start[slot] 대기
        // printf("[MAIN] Waiting for host_can_start[%d] for batch=%d\n", slot, host_batch_idx);
        fflush(stdout);
        while (!host_can_start[slot].load()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        // printf("[MAIN] host_can_start[%d] received, starting Host search for batch=%d\n", slot, host_batch_idx);
        fflush(stdout);

        //hs_pipelining - Host search 수행
        auto t_host_start = std::chrono::high_resolution_clock::now();

        size_t start_idx = host_batch_idx * batch_size;
        size_t batch_n = std::min((size_t)batch_size, n - start_idx);
        const float* batch_x = x + start_idx * d;

        //hs_pipelining - 결과 저장 공간 초기화
        batch_host_indices[slot].clear();
        batch_host_indices[slot].resize(batch_n);
        batch_host_distances[slot].clear();
        batch_host_distances[slot].resize(batch_n);

        //hs_pipelining - Host shard별 검색
        for (const auto& [shard_id, query_list] : batch_host_shard_to_queries[slot]) {
            if (shard_id < 0 || shard_id >= num_shards || !shard_indices[shard_id]) {
                continue;
            }

            const auto& global_ids = partition_to_global[shard_id];
            int shard_size = static_cast<int>(global_ids.size());
            idx_t local_k = std::min(k, (idx_t)shard_size);

            if (local_k == 0) continue;

            //hs_pipelining - 쿼리 벡터 수집
            std::vector<float> shard_queries;
            shard_queries.reserve(query_list.size() * d);
            for (int qi : query_list) {
                const float* query = batch_x + qi * d;
                shard_queries.insert(shard_queries.end(), query, query + d);
            }

            //hs_pipelining - shard 검색
            std::vector<float> local_distances(query_list.size() * local_k);
            std::vector<idx_t> local_indices(query_list.size() * local_k);
            shard_indices[shard_id]->search(query_list.size(), shard_queries.data(),
                                           local_k, local_distances.data(), local_indices.data());

            //hs_pipelining - local ID → global ID 변환 후 결과 저장
            for (size_t i = 0; i < query_list.size(); ++i) {
                int qi = query_list[i];
                auto& indices = batch_host_indices[slot][qi];
                auto& dists = batch_host_distances[slot][qi];

                for (idx_t j = 0; j < local_k; ++j) {
                    idx_t local_id = local_indices[i * local_k + j];
                    if (local_id >= 0 && local_id < shard_size) {
                        indices.push_back(global_ids[local_id]);
                        dists.push_back(local_distances[i * local_k + j]);
                    }
                }
            }
        }

        //hs_pipelining - Host search 타이밍 저장
        auto t_host_end = std::chrono::high_resolution_clock::now();
        batch_timings[host_batch_idx].host_search_ms = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();

        //hs_pipelining - batch_host_done[slot] 설정
        batch_host_done[slot].store(true);
        host_can_start[slot].store(false);  // 다음 사용을 위해 리셋
        // printf("[MAIN] Host search done for batch=%d, slot=%d, host_done=true\n", host_batch_idx, slot);
        fflush(stdout);

        host_batch_idx++;
    }

    //hs_pipelining - Host 마지막 배치 완료 시점 기록
    host_last_done_time = std::chrono::high_resolution_clock::now();

    //hs_pipelining - Schedule thread join (Merge 포함 모든 작업 완료 대기)
    schedule_thread.join();

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

    //hs_pipelining - Idle time 계산
    double idle_ms = 0;
    std::string idle_side = "N/A";
    if (host_last_done_time > dpu_last_recv_time) {
        // DPU가 먼저 끝남 → DPU가 Host를 기다림
        idle_ms = std::chrono::duration<double, std::milli>(host_last_done_time - dpu_last_recv_time).count();
        idle_side = "DPU waiting for Host";
    } else {
        // Host가 먼저 끝남 → Host가 DPU를 기다림
        idle_ms = std::chrono::duration<double, std::milli>(dpu_last_recv_time - host_last_done_time).count();
        idle_side = "Host waiting for DPU";
    }

    printf("\n=== search_pipelined END (total: %.3f ms) ===\n", total_ms);

    //hs_pipelining - 배치별 타이밍 출력
    printf("\n=== Batch-level Timing Breakdown ===\n");
    printf("%-6s %8s %8s %10s %10s %8s %8s %8s\n",
           "Batch", "Meta(ms)", "Send(ms)", "Host(ms)", "Recv(ms)", "Merge(ms)", "Host_Q", "DPU_Q");
    printf("----------------------------------------------------------------------\n");

    double total_meta = 0, total_send = 0, total_host = 0, total_recv = 0, total_merge = 0;
    int total_host_q = 0, total_dpu_q = 0;

    for (int i = 0; i < total_batches; i++) {
        const auto& t = batch_timings[i];
        printf("%-6d %8.3f %8.3f %10.3f %10.3f %8.3f %8d %8d\n",
               i, t.meta_routing_ms, t.dpu_send_ms, t.host_search_ms, t.dpu_recv_ms, t.merge_ms,
               t.host_queries, t.dpu_queries);

        total_meta += t.meta_routing_ms;
        total_send += t.dpu_send_ms;
        total_host += t.host_search_ms;
        total_recv += t.dpu_recv_ms;
        total_merge += t.merge_ms;
        total_host_q += t.host_queries;
        total_dpu_q += t.dpu_queries;
    }

    printf("----------------------------------------------------------------------\n");
    printf("%-6s %8.3f %8.3f %10.3f %10.3f %8.3f %8d %8d\n",
           "SUM", total_meta, total_send, total_host, total_recv, total_merge, total_host_q, total_dpu_q);
    printf("%-6s %8.3f %8.3f %10.3f %10.3f %8.3f\n",
           "AVG", total_meta/total_batches, total_send/total_batches, total_host/total_batches,
           total_recv/total_batches, total_merge/total_batches);

    printf("\n=== Idle Time ===\n");
    printf("Idle: %.3f ms (%s)\n", idle_ms, idle_side.c_str());
    printf("\n");

#else
    // DOCA 없으면 기존 search 사용
    search(n, x, k, distances, labels, nullptr);
#endif
}

#ifdef FAISS_WITH_DOCA

doca_error_t IndexShardBase::init_doca_pool() {
    if (dma_pool.initialized) {
        DOCA_LOG_WARN("DMA memory pool already initialized");
        return DOCA_SUCCESS;
    }

    doca_error_t result;

    //DOCA_LOG_INFO("========================================");
    //DOCA_LOG_INFO("Initializing DMA Memory Pool (16MB, 8 slots)");
    //DOCA_LOG_INFO("========================================");

    // ===================================================================
    // Step 1: Open DOCA device
    // ===================================================================
    //DOCA_LOG_INFO("Step 1: Opening DOCA DMA device...");

    // Open device using same function as dma_copy_core.c (line 350-359)
    result = open_dma_device(&dma_pool.dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to open DOCA DMA device: %s", doca_error_get_descr(result));
        goto cleanup_none;
    }

    // Store device in dma_cfg for compatibility with existing code
    dma_cfg.dev = dma_pool.dev;

    //DOCA_LOG_INFO("  ✓ Device opened successfully");

    // ===================================================================
    // Step 2: Create DOCA MMAP and add device
    // ===================================================================
    //DOCA_LOG_INFO("Step 2: Creating DOCA MMAP object...");

    result = doca_mmap_create(&dma_pool.mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create MMAP: %s", doca_error_get_descr(result));
        goto cleanup_device;
    }

    result = doca_mmap_add_dev(dma_pool.mmap, dma_pool.dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to add device to MMAP: %s", doca_error_get_descr(result));
        goto cleanup_mmap;
    }

    result = doca_mmap_set_permissions(dma_pool.mmap, DOCA_ACCESS_FLAG_PCI_READ_WRITE);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set MMAP permissions: %s", doca_error_get_descr(result));
        goto cleanup_mmap;
    }

    //DOCA_LOG_INFO("  ✓ MMAP created and configured");

    // ===================================================================
    // Step 3: Allocate fixed memory pool (aligned to 4096 bytes)
    // ===================================================================
    //DOCA_LOG_INFO("Step 3: Allocating memory pool...");
    //DOCA_LOG_INFO("  - Total size: %zu bytes (%.1f MB)", dma_pool.total_size, dma_pool.total_size / (1024.0 * 1024.0));
    //DOCA_LOG_INFO("  - Slot size: %zu bytes (%.1f MB)", dma_pool.slot_size, dma_pool.slot_size / (1024.0 * 1024.0));
    //DOCA_LOG_INFO("  - Num slots: %d", dma_pool.num_slots);
    //DOCA_LOG_INFO("  - Alignment: 4096 bytes");

    dma_pool.base_addr = aligned_alloc(4096, dma_pool.total_size);
    if (dma_pool.base_addr == nullptr) {
        DOCA_LOG_ERR("Failed to allocate aligned memory pool (%zu bytes)", dma_pool.total_size);
        goto cleanup_mmap;
    }

    // Zero out the memory pool
    std::memset(dma_pool.base_addr, 0, dma_pool.total_size);

    //DOCA_LOG_INFO("  ✓ Memory pool allocated at address: %p", dma_pool.base_addr);

    // ===================================================================
    // Step 4: Register memory range and start MMAP (triggers mlock)
    // ===================================================================
    //DOCA_LOG_INFO("Step 4: Registering memory range with DOCA MMAP...");
    //DOCA_LOG_INFO("  (This will call mlock() to pin memory - may take ~0.8ms)");

    result = doca_mmap_set_memrange(dma_pool.mmap, dma_pool.base_addr, dma_pool.total_size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set memory range: %s", doca_error_get_descr(result));
        goto cleanup_memory;
    }

    result = doca_mmap_start(dma_pool.mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start MMAP: %s", doca_error_get_descr(result));
        goto cleanup_memory;
    }

    //DOCA_LOG_INFO("  ✓ Memory range registered and MMAP started");

    // ===================================================================
    // Step 5: Export MMAP descriptor and cache it
    // ===================================================================
    //DOCA_LOG_INFO("Step 5: Exporting MMAP descriptor for DPU access...");
    //DOCA_LOG_INFO("  (This creates the serialized descriptor for remote MMAP creation)");

    result = doca_mmap_export_pci(dma_pool.mmap, dma_pool.dev,
                                   &dma_pool.export_desc, &dma_pool.export_desc_len);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to export MMAP: %s", doca_error_get_descr(result));
        goto cleanup_mmap_started;
    }

    //DOCA_LOG_INFO("  ✓ MMAP exported successfully");
    //DOCA_LOG_INFO("  - Export descriptor address: %p", dma_pool.export_desc);
    //DOCA_LOG_INFO("  - Export descriptor length: %zu bytes", dma_pool.export_desc_len);

    // ===================================================================
    // Initialization complete
    // ===================================================================
    dma_pool.initialized = true;
    dma_pool.send_slot = 0;
    dma_pool.recv_slot = 0;

    //DOCA_LOG_INFO("========================================");
    //DOCA_LOG_INFO("✓ DMA Memory Pool Initialized Successfully");
    //DOCA_LOG_INFO("========================================");
    //DOCA_LOG_INFO("Pool configuration:");
    //DOCA_LOG_INFO("  - Base address: %p", dma_pool.base_addr);
    //DOCA_LOG_INFO("  - Total size: %zu bytes (%.1f MB)", dma_pool.total_size, dma_pool.total_size / (1024.0 * 1024.0));
    //DOCA_LOG_INFO("  - Slot size: %zu bytes (%.1f MB)", dma_pool.slot_size, dma_pool.slot_size / (1024.0 * 1024.0));
    //DOCA_LOG_INFO("  - Num slots: %d (circular allocation)", dma_pool.num_slots);
    //DOCA_LOG_INFO("  - Export cached: YES (reused for all requests)");
    //DOCA_LOG_INFO("========================================");
    //DOCA_LOG_INFO("Expected performance gain:");
    //DOCA_LOG_INFO("  - Eliminated per-request overhead: ~2.76ms");
    //DOCA_LOG_INFO("    * Device open: 0.645ms");
    //DOCA_LOG_INFO("    * mlock (memory_alloc_and_populate): 0.810ms");
    //DOCA_LOG_INFO("    * MMAP export: 0.500ms");
    //DOCA_LOG_INFO("    * munlock (doca_mmap_destroy): 0.306ms");
    //DOCA_LOG_INFO("    * DPU create_from_export: 0.500ms");
    //DOCA_LOG_INFO("  - Target runtime overhead: ~0.35ms (memcpy + offset message)");
    //DOCA_LOG_INFO("========================================\n");

    return DOCA_SUCCESS;

// ===================================================================
// Error handling with proper cleanup
// ===================================================================
cleanup_mmap_started:
    // MMAP was started but export failed - need to stop it
    doca_mmap_stop(dma_pool.mmap);

cleanup_memory:
    // Memory was allocated but MMAP registration failed
    if (dma_pool.base_addr != nullptr) {
        free(dma_pool.base_addr);
        dma_pool.base_addr = nullptr;
    }

cleanup_mmap:
    // MMAP was created but configuration failed
    if (dma_pool.mmap != nullptr) {
        doca_mmap_destroy(dma_pool.mmap);
        dma_pool.mmap = nullptr;
    }

cleanup_device:
    // Device was opened but MMAP creation failed
    if (dma_pool.dev != nullptr) {
        doca_dev_close(dma_pool.dev);
        dma_pool.dev = nullptr;
        dma_cfg.dev = nullptr;
    }

cleanup_none:
    DOCA_LOG_ERR("========================================");
    DOCA_LOG_ERR("✗ DMA Memory Pool Initialization Failed");
    DOCA_LOG_ERR("========================================");
    dma_pool.initialized = false;
    return result;
}

void IndexShardBase::cleanup_doca_pool() {
    if (!dma_pool.initialized) {
        //DOCA_LOG_INFO("DMA memory pool not initialized, nothing to cleanup");
        return;
    }

    //DOCA_LOG_INFO("========================================");
    //DOCA_LOG_INFO("Cleaning up DMA Memory Pool...");
    //DOCA_LOG_INFO("========================================");

    // NOTE: export_desc is managed by DOCA and freed automatically when MMAP is destroyed
    // Do NOT manually free it - it will cause double free!
    if (dma_pool.export_desc != nullptr) {
        //DOCA_LOG_INFO("  - Export descriptor will be freed by DOCA when MMAP is destroyed");
        dma_pool.export_desc = nullptr;
        dma_pool.export_desc_len = 0;
    }

    // Stop and destroy MMAP (triggers munlock)
    if (dma_pool.mmap != nullptr) {
        //DOCA_LOG_INFO("  - Stopping MMAP (triggers munlock - may take ~0.3ms)...");
        doca_error_t result = doca_mmap_stop(dma_pool.mmap);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_WARN("Failed to stop MMAP: %s", doca_error_get_descr(result));
        }

        //DOCA_LOG_INFO("  - Destroying MMAP...");
        result = doca_mmap_destroy(dma_pool.mmap);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_WARN("Failed to destroy MMAP: %s", doca_error_get_descr(result));
        }
        dma_pool.mmap = nullptr;
    }

    // Free memory pool
    if (dma_pool.base_addr != nullptr) {
        //DOCA_LOG_INFO("  - Freeing memory pool (%.1f MB at %p)...", dma_pool.total_size / (1024.0 * 1024.0), dma_pool.base_addr);
        free(dma_pool.base_addr);
        dma_pool.base_addr = nullptr;
    }

    // Close device
    if (dma_pool.dev != nullptr) {
        //DOCA_LOG_INFO("  - Closing DOCA device...");
        doca_error_t result = doca_dev_close(dma_pool.dev);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_WARN("Failed to close device: %s", doca_error_get_descr(result));
        }
        dma_pool.dev = nullptr;
        dma_cfg.dev = nullptr;
    }

    dma_pool.initialized = false;
    dma_pool.send_slot = 0;
    dma_pool.recv_slot = 0;

    //DOCA_LOG_INFO("========================================");
    //DOCA_LOG_INFO("✓ DMA Memory Pool Cleanup Complete");
    //DOCA_LOG_INFO("========================================\n");
}

bool IndexShardBase::init_doca_comch() {
    if (doca_initialized) {
        DOCA_LOG_WARN("COMCH already initialized");
        return true;
    }
    
    /* DOCA COMCH - main.cpp와 완전히 동일하게 */
    doca_error_t result;
    struct doca_log_backend *sdk_log;

    // 1) DOCA 로그 백엔드 초기화
    result = doca_log_backend_create_standard();
    if (result != DOCA_SUCCESS) {
        fprintf(stderr, "Failed to create standard log backend: %s\n", doca_error_get_descr(result));
        return false;
    }
    result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
    if (result != DOCA_SUCCESS) {
        fprintf(stderr, "Failed to create SDK log backend: %s\n", doca_error_get_descr(result));
        return false;
    }
    result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_ERROR);
    if (result != DOCA_SUCCESS) {
        fprintf(stderr, "Failed to set SDK log level: %s\n", doca_error_get_descr(result));
        return false;
    }

    // 2) dma_copy_cfg 초기화
    std::memset(&dma_cfg, 0, sizeof(dma_cfg));
    std::strcpy(dma_cfg.cc_dev_pci_addr, "08:00.0");

    // 3) Initialize DMA memory pool BEFORE COMCH
    //    This eliminates per-request MMAP creation/destruction
    //DOCA_LOG_INFO("Initializing DMA memory pool before COMCH...");
    result = init_doca_pool();
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize DMA memory pool: %s", doca_error_get_descr(result));
        return false;
    }

    // 4) COMCH 초기화
    DOCA_LOG_INFO("[INIT] comch_utils_init with dma_cfg=%p", (void*)&dma_cfg);
    result = comch_utils_init(SERVER_NAME,
                            dma_cfg.cc_dev_pci_addr,
                            dma_cfg.cc_dev_rep_pci_addr,
                            &dma_cfg,
                            host_recv_event_cb,
                            nullptr,
                            &comch_cfg);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize COMCH: %s", doca_error_get_descr(result));
        return false;
    }
    //DOCA_LOG_INFO("COMCH initialized successfully on host"); 
    
    doca_initialized = true;
    return true;
}

void IndexShardBase::close_doca_comch() {
    if (!doca_initialized) {
        return;
    }

    // Close COMCH first
    if (comch_cfg) {
        comch_utils_destroy(comch_cfg);
        comch_cfg = nullptr;
    }

    // Then cleanup DMA memory pool
    cleanup_doca_pool();

    doca_initialized = false;
    //DOCA_LOG_INFO("COMCH connection closed");
}

// Batch DMA send function adapted from host_batch
IndexShardBase::DPUSendResult IndexShardBase::send_batch_search_to_dpu(
        const float* queries,
        const std::unordered_map<uint32_t, std::vector<int>>& shard_to_queries,
        int k) const {

    auto t_send_start = std::chrono::high_resolution_clock::now();  // ← 함수 시작 시간

    struct MultiShardReqHeader {
        uint32_t n_shards;
        uint32_t dim;
    };
    struct ShardEntryHeader {
        uint32_t shard_id;
        uint32_t n_queries;
    };
    
    // Build shard list (non-empty shards only)
    std::vector<std::pair<uint32_t, const std::vector<int>*>> shard_list;
    shard_list.reserve(shard_to_queries.size());
    for (const auto& kv : shard_to_queries) {
        if (!kv.second.empty()) {
            shard_list.emplace_back(kv.first, &kv.second);
        }
    }
    
    const uint32_t n_shards = (uint32_t)shard_list.size();
    if (n_shards == 0) {
        auto t_send_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_send_end - t_send_start).count();
        return {DOCA_ERROR_BAD_STATE, elapsed_ms}; // No shards to process
    }
    
    // Calculate total bytes
    size_t total_bytes_req = sizeof(MultiShardReqHeader);
    for (const auto& pr : shard_list) {
        const std::vector<int>& qlist = *pr.second;
        const uint32_t n_q = (uint32_t)qlist.size();
        total_bytes_req += sizeof(ShardEntryHeader);
        total_bytes_req += (size_t)n_q * sizeof(uint32_t);              // qidx array
        total_bytes_req += (size_t)n_q * (size_t)d * sizeof(float);     // vectors
    }

    // Check if request fits in slot
    if (total_bytes_req > dma_pool.slot_size) {
        DOCA_LOG_ERR("Request size (%zu bytes) exceeds slot size (%zu bytes)",
                     total_bytes_req, dma_pool.slot_size);
        auto t_send_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_send_end - t_send_start).count();
        return {DOCA_ERROR_BAD_STATE, elapsed_ms};
    }

    // ===================================================================
    //hs_pipelining - Host→DPU는 slot 0~3 사용 (순환)
    // ===================================================================
    int slot_idx = dma_pool.send_slot++ % 4;  // slot 0~3 for send

    size_t slot_offset = slot_idx * dma_pool.slot_size;
    uint8_t* slot_addr = static_cast<uint8_t*>(dma_pool.base_addr) + slot_offset;

    //DOCA_LOG_INFO("[POOL-SEND] Using slot %d (offset=%zu, size=%zu bytes)", slot_idx, slot_offset, total_bytes_req);

    // 메모리 풀 상태 확인
    //DOCA_LOG_INFO("[POOL] Memory pool status check:");
    //DOCA_LOG_INFO("[POOL]   - base_addr: %p", dma_pool.base_addr);
    //DOCA_LOG_INFO("[POOL]   - slot_addr: %p", slot_addr);
    //DOCA_LOG_INFO("[POOL]   - mmap ptr: %p", dma_pool.mmap);
    //DOCA_LOG_INFO("[POOL]   - initialized: %d", dma_pool.initialized);

    // Package request directly into memory pool slot
    uint8_t* p = slot_addr;

    // Main header
    MultiShardReqHeader mh{ n_shards, (uint32_t)d };
    std::memcpy(p, &mh, sizeof(mh)); p += sizeof(mh);

    //DOCA_LOG_INFO("[DEBUG-SEND] === Sending Request to DPU ===");
    //DOCA_LOG_INFO("[DEBUG-SEND] Main header: n_shards=%u, dim=%u", mh.n_shards, mh.dim);

    // Shard entries
    for (const auto& pr : shard_list) {
        uint32_t sid = pr.first;
        const std::vector<int>& qlist = *pr.second;
        const uint32_t n_q = (uint32_t)qlist.size();

        // Shard entry header
        ShardEntryHeader seh{ sid, n_q };
        std::memcpy(p, &seh, sizeof(seh)); p += sizeof(seh);

        //DOCA_LOG_INFO("[DEBUG-SEND] Shard entry: shard_id=%u, n_queries=%u", seh.shard_id, seh.n_queries);

        // Query indices
        for (uint32_t i = 0; i < n_q; ++i) {
            uint32_t qi32 = (uint32_t)qlist[i];
            std::memcpy(p, &qi32, sizeof(uint32_t));
            p += sizeof(uint32_t);

            //DOCA_LOG_INFO("[DEBUG-SEND]   Query %u: qidx=%u", i, qi32);
        }

        // Query vectors (row-major: n_q × d)
        for (uint32_t i = 0; i < n_q; ++i) {
            int qi = qlist[i];
            const float* qvec = queries + (size_t)qi * (size_t)d;
            std::memcpy(p, qvec, (size_t)d * sizeof(float));

            // Log first 4 dimensions of query vector
            //DOCA_LOG_INFO("[DEBUG-SEND]   Query %u vector[0-3]: %.2f %.2f %.2f %.2f",qi, qvec[0], qvec[1], qvec[2], qvec[3]);

            p += (size_t)d * sizeof(float);
        }
    }

    //DOCA_LOG_INFO("[DEBUG-SEND] === Request packaging complete ===");

    // ===================================================================
    // SEND VIA DMA: Use memory pool instead of file-based approach
    // ===================================================================
    //DOCA_LOG_INFO("[POOL] Sending %zu bytes from slot %d (offset=%zu) to DPU", total_bytes_req, slot_idx, slot_offset);

    // Get max COMCH buffer size (REQUIRED before sending export descriptor!)
    const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer = comch_utils_get_max_buffer_size(comch_cfg);
    if (const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer == 0) {
        DOCA_LOG_ERR("Comch max buffer length is 0");
        auto t_send_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_send_end - t_send_start).count();
        return {DOCA_ERROR_INVALID_VALUE, elapsed_ms};
    }
    //DOCA_LOG_INFO("[POOL] max_comch_buffer = %zu bytes", const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer);

    // Setup DMA configuration to use memory pool
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_mmap = dma_pool.mmap;  // Use pool MMAP
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_buffer = reinterpret_cast<char*>(slot_addr);    // Point to slot
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_size = total_bytes_req;
    const_cast<dma_copy_cfg*>(&dma_cfg)->host_addr = reinterpret_cast<uint8_t*>((uintptr_t)slot_addr);
    const_cast<dma_copy_cfg*>(&dma_cfg)->is_file_found_locally = true;

    // Use cached export descriptor (no need to export again!)
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap = (uint8_t*)dma_pool.export_desc;
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap_len = dma_pool.export_desc_len;

    // Start DMA negotiation
    const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state = COMCH_NEGOTIATING;

    // Send direction message (zero-initialize to prevent garbage in padding bytes)
    struct comch_msg_dma_direction dir_msg = {};
    dir_msg.type = COMCH_MSG_DIRECTION;
    dir_msg.file_in_host = true;  // Send mode
    dir_msg.file_size = htonq(total_bytes_req);  // Convert to network byte order

    doca_error_t ret = comch_utils_send(comch_util_get_connection(comch_cfg), &dir_msg, sizeof(dir_msg));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to send direction message: %s", doca_error_get_descr(ret));
        auto t_send_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_send_end - t_send_start).count();
        return {ret, elapsed_ms};
    }

    // Send export descriptor (with cached descriptor from pool)
    // Allocate buffer for message + variable-sized export descriptor
    size_t exp_msg_size = sizeof(struct comch_msg_dma_export_discriptor) + dma_pool.export_desc_len;
    std::vector<uint8_t> exp_msg_buf(exp_msg_size);

    struct comch_msg_dma_export_discriptor* exp_msg =
        reinterpret_cast<struct comch_msg_dma_export_discriptor*>(exp_msg_buf.data());

    exp_msg->type = COMCH_MSG_EXPORT_DESCRIPTOR;

    // DEBUG: Check slot_addr value before conversion (BATCH PATH)
    //DOCA_LOG_INFO("[DEBUG-BATCH] slot_addr=%p, slot_offset=%zu, slot_idx=%d",  slot_addr, slot_offset, slot_idx);
    //DOCA_LOG_INFO("[DEBUG-BATCH] base_addr=%p, pool.initialized=%d",  dma_pool.base_addr, dma_pool.initialized);

    exp_msg->host_addr = htonq((uintptr_t)slot_addr);  // Convert to network byte order

    // DEBUG: Check what we actually wrote
    //DOCA_LOG_INFO("[DEBUG-BATCH] After htonq: exp_msg->host_addr=0x%016lx (network byte order)",  exp_msg->host_addr);
    //DOCA_LOG_INFO("[DEBUG-BATCH] Original value: 0x%016lx (host byte order)",  (uintptr_t)slot_addr);

    exp_msg->export_desc_len = htonq(dma_pool.export_desc_len);  // Convert to network byte order
    std::memcpy(exp_msg->exported_mmap, dma_pool.export_desc, dma_pool.export_desc_len);

    ret = comch_utils_send(comch_util_get_connection(comch_cfg), exp_msg, exp_msg_size);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to send export descriptor: %s", doca_error_get_descr(ret));
        auto t_send_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_send_end - t_send_start).count();
        return {ret, elapsed_ms};
    }

    // ===================================================================
    // COLLISION HANDLING: Prepare pending_recv BEFORE progress loop
    // If DPU sends DIRECTION(D) during our send, callback will use this
    // Use send slot to determine recv slot: send 0→recv 4, send 1→recv 5, etc.
    // ===================================================================
    int recv_slot_idx = 4 + (slot_idx % 4);  // slot 4~7 for receive (based on send slot)
    size_t recv_slot_offset = recv_slot_idx * dma_pool.slot_size;
    uint8_t* recv_slot_addr = static_cast<uint8_t*>(dma_pool.base_addr) + recv_slot_offset;

    const_cast<dma_copy_cfg*>(&dma_cfg)->pending_recv_buffer = reinterpret_cast<char*>(recv_slot_addr);
    const_cast<dma_copy_cfg*>(&dma_cfg)->pending_recv_slot = recv_slot_idx;
    const_cast<dma_copy_cfg*>(&dma_cfg)->collision_recv_done = false;

    //DOCA_LOG_INFO("[COLLISION] Prepared pending_recv: slot=%d, addr=%p", recv_slot_idx, recv_slot_addr);

    // Wait for DMA completion
    while (const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state == COMCH_NEGOTIATING) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        ret = comch_utils_progress_connection(comch_util_get_connection(comch_cfg));
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Progress failed: %s", doca_error_get_descr(ret));
            break;
        }
    }

    if (const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state == COMCH_ERROR) {
        DOCA_LOG_ERR("DMA send failed (Host→DPU)");
        auto t_send_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_send_end - t_send_start).count();
        return {DOCA_ERROR_BAD_STATE, elapsed_ms};
    }

    // ===================================================================
    // COLLISION HANDLING: If collision occurred, just increment recv_slot
    // Parsing and storing will be done in schedule_thread_func with correct store_slot
    // ===================================================================
    if (const_cast<dma_copy_cfg*>(&dma_cfg)->collision_recv_done) {
        dma_pool.recv_slot++;
        //DOCA_LOG_INFO("[COLLISION] Detected in send_batch_search_to_dpu, recv_slot incremented to %d", dma_pool.recv_slot);
        // collision_recv_done stays true, parsing will be done in schedule_thread_func
    }

    //DOCA_LOG_INFO("[POOL] DMA send completed successfully");

    // Reset comch_state for next operation
    const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state = COMCH_NEGOTIATING;

    // Success: calculate elapsed time and return
    auto t_send_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_send_end - t_send_start).count();
    return {DOCA_SUCCESS, elapsed_ms};
}

// Legacy single query function - OPTIMIZED to use memory pool
doca_error_t IndexShardBase::send_search_to_dpu(
        const float* query,
        const std::vector<uint32_t>& shard_ids,
        int k) const {

    // Package request - DPU expects specific format
    struct MultiShardReqHeader {
        uint32_t n_shards;
        uint32_t dim;
    };
    struct ShardEntryHeader {
        uint32_t shard_id;
        uint32_t n_queries;
    };

    const uint32_t n_sub = shard_ids.size();
    const uint32_t n_queries = 1;  // Single query
    const uint32_t dim_u32 = static_cast<uint32_t>(d);

    // Calculate total size
    size_t total_bytes_req = sizeof(MultiShardReqHeader);
    total_bytes_req += n_sub * (sizeof(ShardEntryHeader) + sizeof(uint32_t) + d * sizeof(float));

    // Check if request fits in slot
    if (total_bytes_req > dma_pool.slot_size) {
        DOCA_LOG_ERR("Request size (%zu bytes) exceeds slot size (%zu bytes)",
                     total_bytes_req, dma_pool.slot_size);
        return DOCA_ERROR_BAD_STATE;
    }

    // ===================================================================
    // USE MEMORY POOL: Legacy single query - use send_slot (slots 0-7, same slot for send/receive)
    // ===================================================================
    int slot_idx = dma_pool.send_slot++ % 8;  // Legacy: use all 8 slots

    size_t slot_offset = slot_idx * dma_pool.slot_size;
    uint8_t* slot_addr = static_cast<uint8_t*>(dma_pool.base_addr) + slot_offset;

    //DOCA_LOG_INFO("[POOL] Using slot %d (offset=%zu, size=%zu bytes)",  slot_idx, slot_offset, total_bytes_req);

    // Package request directly into memory pool slot
    uint8_t* p = slot_addr;

    // Write main header
    MultiShardReqHeader mh = {n_sub, dim_u32};
    std::memcpy(p, &mh, sizeof(mh));
    p += sizeof(mh);

    // Write each shard entry
    for (uint32_t sid : shard_ids) {
        ShardEntryHeader seh = {sid, n_queries};
        std::memcpy(p, &seh, sizeof(seh));
        p += sizeof(seh);

        // Query index (0 for single query)
        uint32_t qidx = 0;
        std::memcpy(p, &qidx, sizeof(qidx));
        p += sizeof(qidx);

        // Query vector
        std::memcpy(p, query, d * sizeof(float));
        p += d * sizeof(float);
    }

    // ===================================================================
    // SEND VIA DMA: Use memory pool instead of file-based approach
    // ===================================================================
    //DOCA_LOG_INFO("[POOL] Sending %zu bytes from slot %d (offset=%zu) to DPU",  total_bytes_req, slot_idx, slot_offset);

    // Get max COMCH buffer size (REQUIRED before sending export descriptor!)
    const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer = comch_utils_get_max_buffer_size(comch_cfg);
    if (const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer == 0) {
        DOCA_LOG_ERR("Comch max buffer length is 0");
        return DOCA_ERROR_INVALID_VALUE;
    }
    //DOCA_LOG_INFO("[POOL] max_comch_buffer = %zu bytes", const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer);

    // Setup DMA configuration to use memory pool
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_mmap = dma_pool.mmap;  // Use pool MMAP
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_buffer = reinterpret_cast<char*>(slot_addr);    // Point to slot
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_size = total_bytes_req;
    const_cast<dma_copy_cfg*>(&dma_cfg)->host_addr = reinterpret_cast<uint8_t*>((uintptr_t)slot_addr);
    const_cast<dma_copy_cfg*>(&dma_cfg)->is_file_found_locally = true;

    // Use cached export descriptor (no need to export again!)
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap = (uint8_t*)dma_pool.export_desc;
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap_len = dma_pool.export_desc_len;

    // Start DMA negotiation
    const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state = COMCH_NEGOTIATING;

    // Send direction message (zero-initialize to prevent garbage in padding bytes)
    struct comch_msg_dma_direction dir_msg = {};
    dir_msg.type = COMCH_MSG_DIRECTION;
    dir_msg.file_in_host = true;  // Send mode
    dir_msg.file_size = htonq(total_bytes_req);  // Convert to network byte order

    doca_error_t ret = comch_utils_send(comch_util_get_connection(comch_cfg), &dir_msg, sizeof(dir_msg));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to send direction message: %s", doca_error_get_descr(ret));
        return ret;
    }

    // Send export descriptor (with cached descriptor from pool)
    size_t exp_msg_size = sizeof(struct comch_msg_dma_export_discriptor) + dma_pool.export_desc_len;
    std::vector<uint8_t> exp_msg_buf(exp_msg_size);

    struct comch_msg_dma_export_discriptor* exp_msg =
        reinterpret_cast<struct comch_msg_dma_export_discriptor*>(exp_msg_buf.data());

    exp_msg->type = COMCH_MSG_EXPORT_DESCRIPTOR;

    // DEBUG: Check slot_addr value before conversion (SINGLE QUERY PATH)
    //DOCA_LOG_INFO("[DEBUG-SINGLE] slot_addr=%p, slot_offset=%zu, slot_idx=%d",  slot_addr, slot_offset, slot_idx);
    //DOCA_LOG_INFO("[DEBUG-SINGLE] base_addr=%p, pool.initialized=%d",  dma_pool.base_addr, dma_pool.initialized);

    exp_msg->host_addr = htonq((uintptr_t)slot_addr);  // Convert to network byte order

    // DEBUG: Check what we actually wrote
    //DOCA_LOG_INFO("[DEBUG-SINGLE] After htonq: exp_msg->host_addr=0x%016lx (network byte order)",  exp_msg->host_addr);
    //DOCA_LOG_INFO("[DEBUG-SINGLE] Original value: 0x%016lx (host byte order)",  (uintptr_t)slot_addr);

    exp_msg->export_desc_len = htonq(dma_pool.export_desc_len);  // Convert to network byte order
    std::memcpy(exp_msg->exported_mmap, dma_pool.export_desc, dma_pool.export_desc_len);

    ret = comch_utils_send(comch_util_get_connection(comch_cfg), exp_msg, exp_msg_size);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to send export descriptor: %s", doca_error_get_descr(ret));
        return ret;
    }

    // Wait for DMA completion
    while (const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state == COMCH_NEGOTIATING) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        ret = comch_utils_progress_connection(comch_util_get_connection(comch_cfg));
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Progress failed: %s", doca_error_get_descr(ret));
            break;
        }
    }

    if (const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state == COMCH_ERROR) {
        DOCA_LOG_ERR("DMA send failed (Host→DPU)");
        return DOCA_ERROR_BAD_STATE;
    }

    //DOCA_LOG_INFO("[POOL] DMA send completed successfully");

    // Reset comch_state for next operation
    const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state = COMCH_NEGOTIATING;

    return DOCA_SUCCESS;
}

// Batch DMA receive function
//hs_pipelining - timeout 파라미터 추가
std::unordered_map<uint32_t, std::vector<IndexShardBase::ResultPair>> IndexShardBase::receive_batch_search_from_dpu(
    std::chrono::high_resolution_clock::time_point* wait_end,
    std::chrono::microseconds timeout) const {
    std::unordered_map<uint32_t, std::vector<ResultPair>> results;

    // Get max buffer size
    const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer = comch_utils_get_max_buffer_size(comch_cfg);
    if (const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer == 0) {
        DOCA_LOG_ERR("Comch max buffer length is 0");
        return results;
    }

    // ===================================================================
    //hs_pipelining - DPU→Host는 slot 4~7 사용 (순환)
    // NOTE: recv_slot은 성공 시에만 증가해야 함 (timeout 시 같은 slot 재사용)
    // ===================================================================
    int slot_idx = 4 + (dma_pool.recv_slot % 4);  // slot 4~7 for receive (증가는 성공 시에만)

    size_t slot_offset = slot_idx * dma_pool.slot_size;
    uint8_t* slot_addr = static_cast<uint8_t*>(dma_pool.base_addr) + slot_offset;

    //DOCA_LOG_INFO("[POOL-RECV] Using slot %d (offset=%zu) for receive", slot_idx, slot_offset);

    // Setup for receive - use memory pool MMAP (no new MMAP creation!)
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_mmap = dma_pool.mmap;  // Reuse pool MMAP
    const_cast<dma_copy_cfg*>(&dma_cfg)->is_file_found_locally = false;
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_buffer = reinterpret_cast<char*>(slot_addr);
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_path[0] = '\0';  // Not using file
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_size = 0;  // CRITICAL: Reset file_size before negotiation

    // OPTIMIZATION: Set cached export descriptor for callback to use (avoids doca_mmap_export_pci call)
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap = (uint8_t*)dma_pool.export_desc;
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap_len = dma_pool.export_desc_len;
    //DOCA_LOG_INFO("[POOL-RECV] Set cached export descriptor (len=%zu) for callback to reuse", dma_pool.export_desc_len);

    // Start negotiation - DPU sends DIRECTION, HOST responds with EXPORT (in callback)
    const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state = COMCH_NEGOTIATING;

    //DOCA_LOG_INFO("[RECV-BATCH] ============================================");
    //DOCA_LOG_INFO("[RECV-BATCH] DPU→HOST Protocol:");
    //DOCA_LOG_INFO("[RECV-BATCH] 1. DPU sends DIRECTION (file_size)");
    //DOCA_LOG_INFO("[RECV-BATCH] 2. HOST callback sends EXPORT");
    //DOCA_LOG_INFO("[RECV-BATCH] 3. DPU receives EXPORT and starts DMA");
    //DOCA_LOG_INFO("[RECV-BATCH] ============================================");
    //DOCA_LOG_INFO("[RECV-BATCH] Waiting for DPU's DIRECTION message...");

    // DO NOT send DIRECTION from HOST in receive mode!
    // The callback (host_process_dma_direction_and_size) will send EXPORT when it receives DPU's DIRECTION

    doca_error_t ret;

    //hs_pipelining - timeout 체크용 시작 시간
    auto poll_start = std::chrono::high_resolution_clock::now();

    // Wait for completion
    //DOCA_LOG_INFO("[RECV-DEBUG] Entering while loop, comch_state=%d, file_size=%zu", (int)const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state, (size_t)const_cast<dma_copy_cfg*>(&dma_cfg)->file_size);

    while (const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state == COMCH_NEGOTIATING) {
        //hs_pipelining - timeout 체크 (timeout > 0이고, DIRECTION 아직 안 받았을 때만)
        //hs_pipelining - file_size > 0이면 DIRECTION 받은 것 → DMA 완료까지 blocking
        if (timeout.count() > 0 && const_cast<dma_copy_cfg*>(&dma_cfg)->file_size == 0) {
            auto elapsed = std::chrono::high_resolution_clock::now() - poll_start;
            if (elapsed > timeout) {
                //DOCA_LOG_INFO("[RECV-DEBUG] Timeout! comch_state=%d, file_size=%zu", (int)const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state, (size_t)const_cast<dma_copy_cfg*>(&dma_cfg)->file_size);
                return results;  // timeout → 빈 map 반환
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        ret = comch_utils_progress_connection(comch_util_get_connection(comch_cfg));
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("progress failed: %s", doca_error_get_descr(ret));
            break;
        }
    }

    //DOCA_LOG_INFO("[RECV-DEBUG] Exited while loop, comch_state=%d, file_size=%zu", (int)const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state, (size_t)const_cast<dma_copy_cfg*>(&dma_cfg)->file_size);

    // Negotiation completed - mark the end of wait time (pure DPU wait, excluding negotiation overhead)
    if (wait_end) {
        *wait_end = std::chrono::high_resolution_clock::now();
    }

    if (const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state == COMCH_ERROR) {
        DOCA_LOG_ERR("DMA failed on DPU→Host path");
        return results;
    }

    //DOCA_LOG_INFO("[POOL-RECV] dma_cfg=%p, DMA receive completed, file_size=%zu bytes, slot_idx=%d, slot_addr=%p",
    //              (void*)&dma_cfg, static_cast<size_t>(const_cast<dma_copy_cfg*>(&dma_cfg)->file_size), slot_idx, slot_addr);

    // Parse results directly from memory pool slot (no file I/O!)
    size_t total_bytes_resp = const_cast<dma_copy_cfg*>(&dma_cfg)->file_size;
    const uint8_t* rbuf = slot_addr;

    // Parse batch results
    // Format: n_blocks | [qidx, n_i, ResultPair[n_i]]...
    const uint8_t* p = rbuf;
    if (total_bytes_resp < sizeof(uint32_t)) {
        DOCA_LOG_ERR("Response too small (file_size=%zu, expected >= 4)", total_bytes_resp);
        return results;
    }

    uint32_t n_blocks = *reinterpret_cast<const uint32_t*>(p);
    p += sizeof(uint32_t);

    // printf("[RECV-PARSE] n_blocks=%u, total_bytes_resp=%zu\n", n_blocks, total_bytes_resp);
    // printf("[RECV-PARSE] DPU returned qidx: ");

    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (p + 2 * sizeof(uint32_t) > rbuf + total_bytes_resp) {
            DOCA_LOG_ERR("Invalid response format");
            break;
        }

        uint32_t qidx = *reinterpret_cast<const uint32_t*>(p);
        p += sizeof(uint32_t);

        uint32_t n_i = *reinterpret_cast<const uint32_t*>(p);
        p += sizeof(uint32_t);

        // printf("%u ", qidx);  // 모든 qidx 출력

        if (p + n_i * sizeof(ResultPair) > rbuf + total_bytes_resp) {
            DOCA_LOG_ERR("Invalid response format");
            break;
        }

        const ResultPair* rp = reinterpret_cast<const ResultPair*>(p);
        auto& vec = results[qidx];
        vec.reserve(vec.size() + n_i);

        for (uint32_t i = 0; i < n_i; ++i) {
            vec.push_back({rp[i].dist, rp[i].id});
        }
        p += n_i * sizeof(ResultPair);
    }
    // printf("\n");  // qidx 출력 후 줄바꿈
    fflush(stdout);

    // Reset DMA state
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap = nullptr;
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap_len = 0;
    const_cast<dma_copy_cfg*>(&dma_cfg)->host_addr = 0;

    // Reset comch_state for next operation
    const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state = COMCH_NEGOTIATING;

    //hs_pipelining - 성공적으로 receive 완료 → recv_slot 증가 (다음 slot 사용)
    dma_pool.recv_slot++;

    return results;
}

// Legacy single query receive function
std::vector<IndexShardBase::ResultPair> IndexShardBase::receive_search_from_dpu() const {
    std::vector<ResultPair> results;

    // Get max buffer size
    const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer = comch_utils_get_max_buffer_size(comch_cfg);
    if (const_cast<dma_copy_cfg*>(&dma_cfg)->max_comch_buffer == 0) {
        DOCA_LOG_ERR("Comch max buffer length is 0");
        return results;
    }

    // ===================================================================
    // USE MEMORY POOL: Reuse same slot as send (no separate receive slots)
    // ===================================================================
    int slot_idx = (dma_pool.send_slot - 1 + 8) % 8;  // Reuse the slot we just sent to

    size_t slot_offset = slot_idx * dma_pool.slot_size;
    uint8_t* slot_addr = static_cast<uint8_t*>(dma_pool.base_addr) + slot_offset;

    //DOCA_LOG_INFO("[POOL-RECV] Using slot %d (offset=%zu) for receive", slot_idx, slot_offset);

    // Setup for receive - use memory pool MMAP (no new MMAP creation!)
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_mmap = dma_pool.mmap;  // Reuse pool MMAP
    const_cast<dma_copy_cfg*>(&dma_cfg)->is_file_found_locally = false;
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_buffer = reinterpret_cast<char*>(slot_addr);
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_path[0] = '\0';  // Not using file
    const_cast<dma_copy_cfg*>(&dma_cfg)->file_size = 0;  // CRITICAL: Reset file_size before negotiation

    // Start negotiation
    const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state = COMCH_NEGOTIATING;
    doca_error_t ret;
    {
        struct comch_msg_dma_direction dir_msg = {.type = COMCH_MSG_DIRECTION};
        dir_msg.file_in_host = false;  // Receive mode
        dir_msg.file_size = 0;  // Initialize to 0 for RECEIVE mode
        ret = comch_utils_send(comch_util_get_connection(comch_cfg), &dir_msg, sizeof(dir_msg));
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("send direction failed: %s", doca_error_get_descr(ret));
            return results;
        }
    }

    // Wait for completion
    while (const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state == COMCH_NEGOTIATING) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        ret = comch_utils_progress_connection(comch_util_get_connection(comch_cfg));
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("progress failed: %s", doca_error_get_descr(ret));
            break;
        }
    }

    if (const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state == COMCH_ERROR) {
        DOCA_LOG_ERR("DMA failed on DPU→Host path");
        return results;
    }

    //DOCA_LOG_INFO("[POOL-RECV] DMA receive completed, size=%zu bytes",  static_cast<size_t>(const_cast<dma_copy_cfg*>(&dma_cfg)->file_size));

    // Parse results directly from memory pool slot (no file I/O!)
    size_t total_bytes_resp = const_cast<dma_copy_cfg*>(&dma_cfg)->file_size;
    const uint8_t* rbuf = slot_addr;
    
    // Parse results - DPU sends blocks with query indices
    // Format: n_blocks | [qidx, n_i, ResultPair[n_i]]...
    const uint8_t* p = rbuf;
    if (total_bytes_resp < sizeof(uint32_t)) {
        DOCA_LOG_ERR("Response too small");
        return results;
    }

    uint32_t n_blocks = *reinterpret_cast<const uint32_t*>(p);
    p += sizeof(uint32_t);

    // For single query, we expect one block with qidx=0
    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (p + 2 * sizeof(uint32_t) > rbuf + total_bytes_resp) {
            DOCA_LOG_ERR("Invalid response format");
            break;
        }

        uint32_t qidx = *reinterpret_cast<const uint32_t*>(p);
        p += sizeof(uint32_t);

        uint32_t n_i = *reinterpret_cast<const uint32_t*>(p);
        p += sizeof(uint32_t);

        if (p + n_i * sizeof(ResultPair) > rbuf + total_bytes_resp) {
            DOCA_LOG_ERR("Invalid response format");
            break;
        }

        if (qidx == 0) {  // Our single query
            const ResultPair* rp = reinterpret_cast<const ResultPair*>(p);
            for (uint32_t i = 0; i < n_i; ++i) {
                results.push_back({rp[i].dist, rp[i].id});
            }
        }
        p += n_i * sizeof(ResultPair);
    }
    
    // Reset DMA state (pyramid.cpp line 351-353)
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap = nullptr;
    const_cast<dma_copy_cfg*>(&dma_cfg)->exported_mmap_len = 0;
    const_cast<dma_copy_cfg*>(&dma_cfg)->host_addr = 0;

    // Reset comch_state to allow next operation (CRITICAL for sequential requests!)
    const_cast<dma_copy_cfg*>(&dma_cfg)->comch_state = COMCH_NEGOTIATING;

    return results;
}

#endif

} // namespace faiss