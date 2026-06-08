/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <queue>
#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>

#ifdef FAISS_WITH_DOCA
#include <doca_argp.h>
#include <doca_comch.h>
#include <doca_log.h>
#include <doca_mmap.h>
#include <doca_error.h>
extern "C" {
#include "/opt/mellanox/doca/applications/common/comch_utils.h"
#include "dma_copy_core_modified.h"
}
#endif

namespace faiss {

/** Base class for sharded indices with meta-level routing
 * 
 * This index partitions data into shards and uses a meta-level
 * HNSW graph to route queries to relevant shards.
 * Future versions will support DPU offloading.
 */
struct IndexShardBase : Index {
    /// Number of shards (default: 2 for host/DPU split)
    int num_shards = 2;
    
    /// Number of shards to search per query (1 or 2)
    int nprobe = 1;
    
    /// HNSW parameters
    int hnsw_M = 32;
    int ef_construction = 40;
    int ef_search = 16;

    /// Boundary-vector replication toggle (runtime; was the PARTITION_NESTED
    /// compile-time macro). When true, boundary vectors detected at build time
    /// are replicated into the neighbouring shard to preserve connectivity.
    bool partition_nested = false;

    /// Meta-level index for routing queries to shards (not exposed to SWIG)
    Index* meta_index = nullptr;
    
    /// Sub-indices for each shard (not exposed to SWIG) 
    std::vector<Index*> shard_indices;
    
    /// Maps shard-local IDs to global IDs
    /// partition_to_global[shard_id][local_id] = global_id
    std::vector<std::vector<idx_t>> partition_to_global;
    
    /// Shard assignment for each vector
    std::vector<int> vector_to_shard;
    
    /// Centroids for routing (num_shards * d)
    std::vector<float> shard_centroids;

    // KH: Random sampling mode for meta index (alternative to centroid mode)
    /// Number of random samples per shard for meta index (0 = use centroid mode)
    int num_samples_per_shard = 0;  // Default: 0 = centroid mode

    // KH: Mapping from meta index ID to shard ID (for sampling mode)
    /// Mapping from meta index ID to shard ID (only used when num_samples_per_shard > 0)
    std::vector<int> meta_id_to_shard;

#ifdef FAISS_WITH_DOCA
    /// Fixed memory pool for DMA operations (eliminates per-request MMAP creation)
    struct DMAMemoryPool {
        // Device and MMAP objects (persistent across requests)
        struct doca_dev* dev = nullptr;              // DOCA device
        struct doca_mmap* mmap = nullptr;            // Persistent MMAP object

        // Memory pool configuration
        void* base_addr = nullptr;                   // Fixed base address (aligned)
        size_t total_size = 16 * 1024 * 1024;       // 16MB total pool size
        size_t slot_size = 2 * 1024 * 1024;         // 2MB per slot
        int num_slots = 8;                           // 8 slots total
        //hs_pipelining - Host→DPU는 slot 0~3, DPU→Host는 slot 4~7
        int send_slot = 0;                           // Host→DPU slot index (0~3, circular)
        int recv_slot = 0;                           // DPU→Host slot index (4~7, offset 4 added in code)

        // Export descriptor (cached and reused)
        const void* export_desc = nullptr;           // Exported MMAP descriptor
        size_t export_desc_len = 0;                  // Descriptor length

        // Initialization status
        bool initialized = false;
    };

    /// DMA memory pool for send operations (mutable for atomic updates in const methods)
    mutable DMAMemoryPool dma_pool;

    /// DOCA configuration for DPU communication
    struct comch_cfg* comch_cfg = nullptr;
    struct dma_copy_cfg dma_cfg;

    /// DOCA initialization status
    bool doca_initialized = false;

    /// Result pair structure from pyramid.cpp (moved here for forward declaration)
    struct ResultPair {
        float dist;
        uint32_t id;
    };

    //hs_pipelining - 동시 관리 가능한 최대 배치 수
    static constexpr int MAX_INFLIGHT_BATCHES = 4;
    static constexpr int MAX_TOTAL_BATCHES = 1024;

    //hs_pipelining - 배치별 타이밍 측정 구조체
    struct BatchTiming {
        double meta_routing_ms = 0;    // Meta search (routing)
        double dpu_send_ms = 0;        // DPU send
        double host_search_ms = 0;     // Host search
        double dpu_recv_ms = 0;        // DPU receive (actual recv time)
        double merge_ms = 0;           // Final merge
        int host_queries = 0;          // Host에 할당된 쿼리 수
        int dpu_queries = 0;           // DPU에 할당된 쿼리 수
    };
    mutable BatchTiming batch_timings[MAX_TOTAL_BATCHES];

    //hs_pipelining - Idle time 측정용
    mutable std::chrono::high_resolution_clock::time_point host_last_done_time;
    mutable std::chrono::high_resolution_clock::time_point dpu_last_recv_time;

    //hs_pipelining - 배치별 상태 플래그 (slot 기반, 4개)
    mutable std::atomic<bool> batch_host_done[MAX_INFLIGHT_BATCHES];
    mutable std::atomic<bool> host_can_start[MAX_INFLIGHT_BATCHES];

    //hs_pipelining - DPU recv 필요 여부 (batch 기반, 64개) - slot 오염 방지
    mutable bool batch_needs_dpu_recv[MAX_TOTAL_BATCHES];

    //hs_pipelining - 배치별 결과 저장 공간 (배열)
    mutable std::vector<std::vector<idx_t>> batch_host_indices[MAX_INFLIGHT_BATCHES];
    mutable std::vector<std::vector<float>> batch_host_distances[MAX_INFLIGHT_BATCHES];
    mutable std::unordered_map<uint32_t, std::vector<ResultPair>> batch_dpu_results[MAX_INFLIGHT_BATCHES];

    //hs_pipelining - 배치별 shard-to-queries 매핑 (배열)
    mutable std::unordered_map<int, std::vector<int>> batch_host_shard_to_queries[MAX_INFLIGHT_BATCHES];
    mutable std::unordered_map<uint32_t, std::vector<int>> batch_dpu_shard_to_queries[MAX_INFLIGHT_BATCHES];
#endif
    
    /// Constructor
    explicit IndexShardBase(int d = 0, MetricType metric = METRIC_L2);
    
    /// Destructor
    ~IndexShardBase() override;
    
    /// Not implemented for sharded indices
    void add(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
    
    /// Search k nearest neighbors
    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params = nullptr) const override;

    //hs_pipelining - 파이프라인 검색 (batch_queue에서 배치를 꺼내서 Host/DPU 병렬 처리)
    void search_pipelined(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int batch_size,
        std::queue<size_t>& batch_queue) const;

    /// Train on dataset (must be overridden by subclasses)
    void train(idx_t n, const float* x) override = 0;
    
    /// Reset index
    void reset() override;
    
    /// Set number of shards to probe during search
    void set_nprobe(int n) { nprobe = n; }
    int get_nprobe() const { return nprobe; }

#ifdef FAISS_WITH_DOCA
    /// Initialize DOCA communication with DPU
    bool init_doca_comch();

    /// Close DOCA communication
    void close_doca_comch();

    /// Initialize DMA memory pool (called once at startup)
    doca_error_t init_doca_pool();

    /// Cleanup DMA memory pool (called at shutdown)
    void cleanup_doca_pool();

    /// DPU send timing result
    struct DPUSendResult {
        doca_error_t error;
        double send_time_ms;
    };

private:
    /// Send batch search request to DPU via DMA
    DPUSendResult send_batch_search_to_dpu(
        const float* queries,
        const std::unordered_map<uint32_t, std::vector<int>>& shard_to_queries,
        int k) const;
    
    /// Receive batch search results from DPU via DMA
    //hs_pipelining - timeout 파라미터 추가 (0 = 무한 대기, >0 = timeout)
    std::unordered_map<uint32_t, std::vector<ResultPair>> receive_batch_search_from_dpu(
        std::chrono::high_resolution_clock::time_point* wait_end = nullptr,
        std::chrono::microseconds timeout = std::chrono::microseconds(0)) const;
    
    /// Send search request to DPU via DMA (legacy single query)
    doca_error_t send_search_to_dpu(
        const float* query,
        const std::vector<uint32_t>& shard_ids,
        int k) const;
    
    /// Receive search results from DPU via DMA (legacy single query)
    std::vector<ResultPair> receive_search_from_dpu() const;

    //hs_pipelining - 파이프라인용 헬퍼 함수
    void do_meta_routing(int slot, const float* x, size_t start_idx,
                         size_t batch_n, int effective_nprobe) const;

    void do_dpu_send(int slot, int batch_idx, const float* x, size_t start_idx, idx_t k) const;

    bool do_dpu_receive(int slot, int batch_idx) const;

    void do_merge(int slot, int batch_idx, idx_t k, float* distances, idx_t* labels,
                  size_t start_idx, size_t batch_n) const;

    void schedule_thread_func(int total_batches, idx_t n, const float* x,
                              int batch_size, idx_t k, int effective_nprobe,
                              float* distances, idx_t* labels,
                              std::queue<size_t>& batch_queue) const;

public:
#endif
    
protected:
    /// Batch search implementation
    void search_batch(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int effective_nprobe) const;
        
    /// Build meta-level index from shard centroids
    void build_meta_index();
    
    /// Build sub-indices for each shard
    void build_shard_indices(const float* x, idx_t n, const std::vector<int>& assignments);

    /// Build sub-indices for each shard with boundary vectors added before HNSW build
    void build_shard_indices_with_boundaries(const float* x, idx_t n,
                                             const std::vector<int>& assignments,
                                             const std::vector<idx_t>& boundary_vectors);
    
    /// Compute centroids for each shard
    void compute_shard_centroids(const float* x, idx_t n, const std::vector<int>& assignments);
    
    /// Add vectors to existing trained shards
    void add_to_existing_shards(idx_t n, const float* x, const idx_t* xids);
    
    /// Route query to top nprobe shards
    std::vector<int> route_query(const float* query) const;
    
    /// Search within a specific shard
    void search_shard(
        int shard_id,
        const float* query,
        int k,
        std::vector<idx_t>& indices,
        std::vector<float>& distances) const;
    
    /// Merge results from multiple shards
    void merge_shard_results(
        const std::vector<std::vector<idx_t>>& shard_indices,
        const std::vector<std::vector<float>>& shard_distances,
        idx_t k,
        idx_t* labels,
        float* distances) const;
};

/** Parameters for sharded index search */
struct SearchParametersShard : SearchParameters {
    int nprobe = 1;  ///< number of shards to probe
    
    virtual ~SearchParametersShard() {}
};

} // namespace faiss