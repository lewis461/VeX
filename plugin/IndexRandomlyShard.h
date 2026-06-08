/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "IndexShardBase.h"
#include <random>

namespace faiss {

/** Index with random data partitioning
 * 
 * This index randomly distributes vectors across shards and uses
 * a meta-level HNSW to route queries. This serves as a baseline
 * to demonstrate worst-case routing behavior.
 */
struct IndexRandomlyShard : IndexShardBase {
    /// Random number generator seed
    uint32_t random_seed;
    
    /// Constructor
    explicit IndexRandomlyShard(int d = 0, MetricType metric = METRIC_L2);
    
    /// Train on dataset with random partitioning
    void train(idx_t n, const float* x) override;
    
private:
    /// Perform random partitioning of data
    std::vector<int> random_partition(idx_t n);
};

} // namespace faiss