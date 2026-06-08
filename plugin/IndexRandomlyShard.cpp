/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "IndexRandomlyShard.h"

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>

namespace faiss {

IndexRandomlyShard::IndexRandomlyShard(int d, MetricType metric)
        : IndexShardBase(d, metric), random_seed(42) {}

void IndexRandomlyShard::train(idx_t n, const float* x) {
    printf("IndexRandomlyShard::train called with n=%zd, d=%d, num_shards=%d, verbose=%d\n", 
           n, d, num_shards, verbose);
    
    FAISS_THROW_IF_NOT_MSG(n > 0, "Cannot train on empty dataset");
    FAISS_THROW_IF_NOT_MSG(x, "Training data is null");
    FAISS_THROW_IF_NOT_MSG(num_shards > 0, "num_shards must be set before training");
    
    if (verbose) {
        printf("IndexRandomlyShard: training on %zd vectors with %d shards\n", 
               n, num_shards);
    }
    
    // Perform random partitioning
    std::vector<int> assignments = random_partition(n);
    
    // Compute centroids for each shard
    compute_shard_centroids(x, n, assignments);
    
    // Build meta-level HNSW index
    build_meta_index();
    
    // Build individual shard indices
    build_shard_indices(x, n, assignments);
    
    ntotal = n;
    is_trained = true;
    
    if (verbose) {
        printf("IndexRandomlyShard: training completed\n");
    }
}

std::vector<int> IndexRandomlyShard::random_partition(idx_t n) {
    if (verbose) {
        printf("IndexRandomlyShard: performing random partitioning\n");
    }
    
    std::vector<int> assignments(n);
    std::mt19937 rng(random_seed);
    std::uniform_int_distribution<int> dist(0, num_shards - 1);
    
    for (idx_t i = 0; i < n; i++) {
        assignments[i] = dist(rng);
    }
    
    // Log distribution
    if (verbose) {
        std::vector<int> counts(num_shards, 0);
        for (int shard_id : assignments) {
            counts[shard_id]++;
        }
        
        printf("IndexRandomlyShard: shard distribution:");
        for (int i = 0; i < num_shards; i++) {
            printf(" shard_%d=%d", i, counts[i]);
        }
        printf("\n");
    }
    
    return assignments;
}

} // namespace faiss