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

/** Index with k-means data partitioning
 *
 * This index uses k-means clustering to partition data into shards
 * and builds a meta-level HNSW from cluster centroids for routing.
 * This should provide better locality than random partitioning.
 */
struct IndexKmeansShard : IndexShardBase {
    /// Clustering parameters
    int niter;

    /// KH: Boundary vectors (used in PARTITION_NESTED mode)
    /// These vectors changed assignment during late k-means iterations
    /// and will be added to both adjacent shards
    std::vector<idx_t> boundary_vectors;

    /// Constructor
    explicit IndexKmeansShard(int d = 0, MetricType metric = METRIC_L2);

    /// Train on dataset with k-means partitioning
    void train(idx_t n, const float* x) override;

private:
    /// Perform k-means partitioning of data
    std::vector<int> kmeans_partition(idx_t n, const float* x);
};

} // namespace faiss