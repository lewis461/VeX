/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "IndexShardBase.h"
#include <faiss/Clustering.h>

namespace faiss {

/** Index with fine-grained k-means clustering followed by greedy merging
 *
 * This index performs k-means with many clusters (e.g., 1000), then
 * merges them into shards using a greedy algorithm that considers both
 * semantic similarity and partition size constraints.
 *
 * Advantages over two-stage k-means:
 * - Better preservation of semantic locality
 * - Minimized graph edge cuts at partition boundaries
 * - More accurate router graph from fine-grained centroids
 */
/// Cluster-merging strategy (runtime-selectable; see merge_mode below).
enum class MergeMode {
    PCA,            ///< PCA-based bisection (best locality) -- VeX default
    GREEDY_CENTROID,///< greedy by distance to partition centroid (updated per step)
    GREEDY_MIN_DIST ///< greedy by min distance to any cluster in the shard
};

struct IndexKmeansShardFinegrained : IndexShardBase {
    /// K-means clustering parameters
    int niter;                    // Number of k-means iterations
    int num_fine_clusters;        // Number of fine-grained clusters (e.g., 1000)

    /// Cluster merging parameters
    float partition_ratio;        // Target ratio for partition 0 (e.g., 0.5 for 50:50)

    /// VeX design toggles (runtime; previously compile-time macros)
    MergeMode merge_mode = MergeMode::PCA;   ///< PCA vs greedy merging
    /// Boundary-vector replication: enabled via base class `partition_nested`.
    /// A vector is a boundary vector if dist_2nd / dist_1st < boundary_tau
    /// (only used when partition_nested == true).
    float boundary_tau = 1.01f;

    /// KH: Partition data save options
    std::string partition_save_path;  // If non-empty, save partitioned data to fvecs files
    std::string partition_base_name;  // Base name for partition files (e.g., "TriviaQA")

    /// Boundary vectors (populated when partition_nested == true)
    std::vector<idx_t> boundary_vectors;

    /// Constructor
    explicit IndexKmeansShardFinegrained(int d = 0, MetricType metric = METRIC_L2);

    /// Train on dataset with fine-grained k-means + greedy merging
    void train(idx_t n, const float* x) override;

private:
    /// Perform fine-grained k-means clustering
    /// Returns: cluster assignments for each vector
    std::vector<int> perform_finegrained_clustering(idx_t n, const float* x);

    /// Assign fine-grained clusters to shards using greedy algorithm
    /// Returns: cluster_id -> shard_id mapping
    std::vector<int> assign_clusters_to_shards_greedy(
        const std::vector<float>& centroids,
        const std::vector<int>& cluster_sizes);

    /// Compute L2 distance between two centroids
    float compute_centroid_distance(
        const float* centroid1,
        const float* centroid2) const;

    /// Find minimum distance from a cluster to any cluster in a shard
    float min_distance_to_shard(
        int cluster_id,
        int shard_id,
        const std::vector<int>& cluster_to_shard,
        const std::vector<float>& centroids) const;

    /// Assign clusters using greedy algorithm with partition centroid updates
    /// (MERGING_MODE_GREEDY_CENTROID)
    std::vector<int> assign_clusters_to_shards_greedy_centroid(
        const std::vector<float>& centroids,
        const std::vector<int>& cluster_sizes);

    /// Assign clusters using PCA-based bisection
    /// (MERGING_MODE_PCA)
    std::vector<int> assign_clusters_to_shards_pca(
        const std::vector<float>& centroids,
        const std::vector<int>& cluster_sizes);

    /// Compute partition centroid from assigned clusters
    void compute_partition_centroid(
        int shard_id,
        const std::vector<int>& cluster_to_shard,
        const std::vector<float>& centroids,
        const std::vector<int>& cluster_sizes,
        float* partition_centroid) const;

    /// Compute distance from cluster to partition centroid
    float distance_to_partition_centroid(
        int cluster_id,
        const std::vector<float>& centroids,
        const float* partition_centroid) const;
};

} // namespace faiss
