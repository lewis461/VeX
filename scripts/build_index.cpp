// VeX index builder (offline).
//
// Builds a semantic-aware sharded HNSW index: fine-grained K-means ->
// PCA/greedy merge -> per-partition HNSW + router graph (+ optional boundary
// vector replication). All VeX design options are runtime flags.
//
// Usage:
//   build_index -base data.fvecs -output ./idx \
//               [-merge pca|greedy|greedy_min] [-nested on|off] [-tau 1.01] \
//               [-clusters 1000] [-ratio 0.5] [-shards 2] [-M 32] [-ef 40] \
//               [-niter 25]
#include <sys/stat.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <IndexKmeansShardFinegrained.h>
#include <faiss/index_io.h>

static std::vector<float> read_fvecs(const std::string& fn, int& n, int& d) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << fn << "\n"; exit(1); }
    f.read(reinterpret_cast<char*>(&d), sizeof(int));
    f.seekg(0, std::ios::end);
    n = f.tellg() / (long long)((1 + d) * sizeof(float));
    std::vector<float> data((size_t)n * d);
    f.seekg(0, std::ios::beg);
    for (int i = 0; i < n; i++) {
        int dd; f.read(reinterpret_cast<char*>(&dd), sizeof(int));
        f.read(reinterpret_cast<char*>(&data[(size_t)i * d]), (long long)d * sizeof(float));
    }
    return data;
}

// Save: meta.index, subIndex/sub_*.index, partitions.bin (+ meta_id_to_shard).
static void save_index(const faiss::IndexShardBase& idx, const std::string& dir) {
    std::string cmd = "mkdir -p " + dir + "/subIndex";
    if (system(cmd.c_str()) != 0) { std::cerr << "mkdir failed\n"; exit(1); }
    for (int i = 0; i < idx.num_shards; i++) {
        if (idx.shard_indices[i])
            faiss::write_index(idx.shard_indices[i],
                               (dir + "/subIndex/sub_" + std::to_string(i) + ".index").c_str());
    }
    if (idx.meta_index)
        faiss::write_index(idx.meta_index, (dir + "/meta.index").c_str());

    std::ofstream o(dir + "/partitions.bin", std::ios::binary);
    faiss::idx_t total = idx.ntotal; int dim = idx.d, ns = idx.num_shards;
    int num_samples = idx.num_samples_per_shard;
    o.write((char*)&total, sizeof(total));
    o.write((char*)&dim, sizeof(dim));
    o.write((char*)&ns, sizeof(ns));
    o.write((char*)&num_samples, sizeof(num_samples));
    size_t meta_id_size = idx.meta_id_to_shard.size();
    o.write((char*)&meta_id_size, sizeof(meta_id_size));
    if (meta_id_size) o.write((char*)idx.meta_id_to_shard.data(), meta_id_size * sizeof(int));
    for (const auto& p : idx.partition_to_global) {
        size_t s = p.size();
        o.write((char*)&s, sizeof(s));
        if (s) o.write((char*)p.data(), s * sizeof(faiss::idx_t));
    }
}

int main(int argc, char** argv) {
    std::string base, output = "./vex_index", merge = "pca", nested = "off";
    int clusters = 1000, shards = 2, M = 32, ef = 40, niter = 25;
    float ratio = 0.5f, tau = 1.01f;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-base") base = next();
        else if (a == "-output") output = next();
        else if (a == "-merge") merge = next();
        else if (a == "-nested") nested = next();
        else if (a == "-tau") tau = std::stof(next());
        else if (a == "-clusters") clusters = std::stoi(next());
        else if (a == "-ratio") ratio = std::stof(next());
        else if (a == "-shards") shards = std::stoi(next());
        else if (a == "-M") M = std::stoi(next());
        else if (a == "-ef") ef = std::stoi(next());
        else if (a == "-niter") niter = std::stoi(next());
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: build_index -base <f.fvecs> -output <dir>\n"
                         "  -merge pca|greedy|greedy_min  (default pca)\n"
                         "  -nested on|off  -tau <float>  (boundary replication)\n"
                         "  -clusters <k> -ratio <0..1> -shards 2 -M -ef -niter\n";
            return 0;
        } else { std::cerr << "Unknown arg: " << a << "\n"; return 1; }
    }
    if (base.empty()) { std::cerr << "-base required\n"; return 1; }

    faiss::MergeMode mm = faiss::MergeMode::PCA;
    if (merge == "greedy") mm = faiss::MergeMode::GREEDY_CENTROID;
    else if (merge == "greedy_min") mm = faiss::MergeMode::GREEDY_MIN_DIST;
    else if (merge != "pca") { std::cerr << "bad -merge\n"; return 1; }

    int n, d;
    std::cout << "Loading " << base << " ...\n";
    auto x = read_fvecs(base, n, d);
    std::cout << "n=" << n << " d=" << d << "\n";

    auto* index = new faiss::IndexKmeansShardFinegrained(d, faiss::METRIC_L2);
    index->num_shards = shards;
    index->num_fine_clusters = clusters;
    index->partition_ratio = ratio;
    index->niter = niter;
    index->hnsw_M = M;
    index->ef_construction = ef;
    index->ef_search = 64;
    index->merge_mode = mm;
    index->partition_nested = (nested == "on");
    index->boundary_tau = tau;
    index->verbose = true;

    index->train(n, x.data());
    save_index(*index, output);
    std::cout << "\nVeX index saved to " << output << "\n";
    delete index;
    return 0;
}
