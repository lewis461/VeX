// VeX search / evaluation driver.
//
// Loads a saved VeX index, runs k-NN search and reports recall@k and latency.
// Exact ground truth is computed by brute force (IndexFlatL2) on the same base
// the index was built from, so it works for any subset.
//
// Host-only by default (both shards in host memory). With a DOCA build, -dpu
// places shard 1 on the BlueField DPU and -pipeline overlaps host/DPU search.
//
// Usage:
//   search_index -index ./idx -base data.fvecs -query q.fvecs \
//                [-nq 1000] [-k 10] [-nprobe 1] [-pipeline] [-dpu] [-bs 1]
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>

#include <IndexKmeansShard.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

static std::vector<float> read_fvecs(const std::string& fn, int& n, int& d) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) { std::cerr << "open fail: " << fn << "\n"; exit(1); }
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

static faiss::IndexShardBase* load_index(const std::string& dir, int d, int nprobe) {
    auto* idx = new faiss::IndexKmeansShard(d, faiss::METRIC_L2);
    idx->num_shards = 2; idx->nprobe = nprobe; idx->verbose = false;
    idx->meta_index = faiss::read_index((dir + "/meta.index").c_str());
    idx->shard_indices.resize(2);
    idx->shard_indices[0] = faiss::read_index((dir + "/subIndex/sub_0.index").c_str());
    idx->shard_indices[1] = faiss::read_index((dir + "/subIndex/sub_1.index").c_str());
    idx->is_trained = true;

    std::ifstream ifs(dir + "/partitions.bin", std::ios::binary);
    if (!ifs) { std::cerr << "no partitions.bin\n"; exit(1); }
    faiss::idx_t total; int dimc, nsc;
    ifs.read((char*)&total, sizeof(total));
    ifs.read((char*)&dimc, sizeof(dimc));
    ifs.read((char*)&nsc, sizeof(nsc));
    int num_samples = 0; size_t mlen = 0;
    ifs.read((char*)&num_samples, sizeof(num_samples));
    ifs.read((char*)&mlen, sizeof(mlen));
    idx->num_samples_per_shard = num_samples;
    if (mlen) { idx->meta_id_to_shard.resize(mlen);
                ifs.read((char*)idx->meta_id_to_shard.data(), mlen * sizeof(int)); }
    idx->partition_to_global.resize(2);
    for (int i = 0; i < 2; i++) {
        size_t s; ifs.read((char*)&s, sizeof(s));
        idx->partition_to_global[i].resize(s);
        if (s) ifs.read((char*)idx->partition_to_global[i].data(), s * sizeof(faiss::idx_t));
    }
    idx->ntotal = 0;
    for (int i = 0; i < 2; i++) if (idx->shard_indices[i]) idx->ntotal += idx->shard_indices[i]->ntotal;
    return idx;
}

int main(int argc, char** argv) {
    std::string index_dir, base, query;
    int nq = 1000, k = 10, nprobe = 1, bs = 1;
    bool pipeline = false, use_dpu = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "-index") index_dir = nx();
        else if (a == "-base") base = nx();
        else if (a == "-query") query = nx();
        else if (a == "-nq") nq = std::stoi(nx());
        else if (a == "-k") k = std::stoi(nx());
        else if (a == "-nprobe") nprobe = std::stoi(nx());
        else if (a == "-bs") bs = std::stoi(nx());
        else if (a == "-pipeline") pipeline = true;
        else if (a == "-dpu") use_dpu = true;
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: search_index -index <dir> -base <f.fvecs> -query <q.fvecs>\n"
                         "  -nq -k -nprobe(1|2) -bs -pipeline -dpu\n";
            return 0;
        } else { std::cerr << "Unknown arg: " << a << "\n"; return 1; }
    }
    if (index_dir.empty() || base.empty() || query.empty()) {
        std::cerr << "-index, -base, -query are required\n"; return 1;
    }
#ifndef FAISS_WITH_DOCA
    if (use_dpu) { std::cerr << "-dpu requires a DOCA build (FAISS_WITH_DOCA)\n"; return 1; }
#endif

    int dq, db, nqf, nb;
    auto q = read_fvecs(query, nqf, dq);
    if (nq > nqf) nq = nqf;
    auto xb = read_fvecs(base, nb, db);
    if (db != dq) { std::cerr << "dim mismatch base/query\n"; return 1; }
    std::cout << "nb=" << nb << " d=" << db << " nq=" << nq << " k=" << k
              << " nprobe=" << nprobe << (pipeline ? " [pipeline]" : "")
              << (use_dpu ? " [dpu]" : " [host-only]") << "\n";

    // exact ground truth on this base
    std::cout << "Computing exact GT (brute force)...\n";
    faiss::IndexFlatL2 flat(db); flat.add(nb, xb.data());
    std::vector<faiss::idx_t> gt((size_t)nq * k); std::vector<float> gd((size_t)nq * k);
    flat.search(nq, q.data(), k, gd.data(), gt.data());

    faiss::IndexShardBase* idx = load_index(index_dir, db, nprobe);
#ifdef FAISS_WITH_DOCA
    if (use_dpu) { idx->shard_indices[1] = nullptr; idx->init_doca_comch(); }
#endif

    std::vector<faiss::idx_t> lab((size_t)nq * k); std::vector<float> dis((size_t)nq * k);
    auto t0 = std::chrono::high_resolution_clock::now();
    if (pipeline) {
        std::queue<size_t> bq;
        for (int s = 0; s < nq; s += bs) bq.push(s);
        idx->search_pipelined(nq, q.data(), k, dis.data(), lab.data(), bs, bq);
    } else {
        idx->search(nq, q.data(), k, dis.data(), lab.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    long correct = 0;
    for (int qi = 0; qi < nq; qi++) {
        std::unordered_set<faiss::idx_t> truth(gt.begin() + (size_t)qi * k,
                                               gt.begin() + (size_t)(qi + 1) * k);
        for (int i = 0; i < k; i++)
            if (truth.count(lab[(size_t)qi * k + i])) correct++;
    }
    double recall = (double)correct / ((double)nq * k);
    std::cout << std::fixed << std::setprecision(4)
              << "recall@" << k << " = " << recall * 100 << "%\n"
              << "latency = " << ms / nq << " ms/query, QPS = " << (nq * 1000.0 / ms) << "\n";
#ifdef FAISS_WITH_DOCA
    if (use_dpu) idx->close_doca_comch();
#endif
    delete idx;
    return 0;
}
