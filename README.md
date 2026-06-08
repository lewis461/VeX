# VeX — Scaling HNSW Vector Search with DPU Memory and Parallelism

**Paper:** [VeX: Scaling HNSW-Based Vector Search with DPU Memory and Parallelism (CCGRID 2026)](https://discos.sogang.ac.kr/file/2026/intl_conf/CCGRID_2026_K_Kim.pdf)

VeX is a host–DPU integrated vector search system that uses a SmartNIC/DPU as
both an extended memory tier and a parallel search engine for HNSW. It (i)
partitions the dataset while preserving semantic structure and places
independent HNSW sub-indices on the host and DPU, (ii) moves data over a
DMA-based PCIe path with pre-registered ring buffers, and (iii) overlaps search,
communication, and aggregation with a heterogeneity-aware pipeline.

This repository packages VeX as a **FAISS plugin** plus two small drivers.

> Scaffold README — expand with paper abstract, figures, citation, and results.

## Repository layout

```
plugin/                     VeX implementation (compiled into libfaiss)
  IndexShardBase.{h,cpp}        sharded index: router, host search, DPU pipelined search
  IndexKmeansShardFinegrained.* fine-grained k-means + PCA/greedy merge (+ boundary replication)
  IndexKmeansShard.*            simple 2-way k-means partitioning (baseline)
  IndexRandomlyShard.*          random partitioning (baseline)
  dma_copy_core_modified.{c,h}  DOCA DMA glue (DPU path only)
  CMakeLists.txt
scripts/
  build_index.cpp / run_build.sh    one index-build driver (all options are flags)
  search_index.cpp / run_search.sh  one search/eval driver
build.sh                    integrate plugin into FAISS, build lib + drivers
```

## Build

Host-only (default):

```bash
# FAISS_SRC may point to an existing FAISS checkout; otherwise it is cloned.
FAISS_SRC=/path/to/faiss ./build.sh
```

DPU / pipelined path (requires NVIDIA DOCA + BlueField DPU):

```bash
WITH_DOCA=1 FAISS_SRC=/path/to/faiss ./build.sh
```

Binaries are written to `bin/`.

## Index build (design options are runtime flags)

```bash
./scripts/run_build.sh -base data.fvecs -output ./idx \
    -merge pca|greedy|greedy_min \   # cluster-merge strategy (default pca)
    -nested on|off  -tau 1.01 \      # boundary-vector replication + threshold
    -clusters 1000 -ratio 0.5 \      # # fine clusters, host:dpu split
    -shards 2 -M 32 -ef 40 -niter 25
```

- `-merge pca` — PCA-guided merge (best locality; VeX default).
- `-merge greedy` / `greedy_min` — greedy merge variants (baselines).
- `-nested on -tau <t>` — replicate boundary vectors (`dist2/dist1 < t`) into the
  neighbouring partition to preserve connectivity.

## Search / evaluation

```bash
./scripts/run_search.sh -index ./idx -base data.fvecs -query q.fvecs \
    -nq 1000 -k 10 -nprobe 1        # host-only; exact GT computed by brute force
# DOCA build only:
./scripts/run_search.sh -index ./idx -base data.fvecs -query q.fvecs -dpu -pipeline -bs 16
```

- `-nprobe 1` routes each query to a single partition (the VeX design point).
- `-dpu` places shard 1 on the DPU; `-pipeline` overlaps host/DPU search.

## Data format

`.fvecs` — for each vector: `int32 dim` followed by `dim` × `float32`.

## License

MIT (see `LICENSE`); inherits FAISS's license for the vendored build.
