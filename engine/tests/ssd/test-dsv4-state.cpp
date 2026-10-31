// SPDX-License-Identifier: MIT
// Model-free roundtrip through the real DSv4 serializer, including both ratio families.
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static void require(bool ok) { if (!ok) { throw std::runtime_error("DSv4 state regression"); } }
struct writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * p, size_t n) override {
        const auto * b = static_cast<const uint8_t *>(p);
        bytes.insert(bytes.end(), b, b + n);
    }
    void write_tensor(const ggml_tensor * t, size_t off, size_t n) override {
        const size_t start = bytes.size(); bytes.resize(start + n);
        ggml_backend_tensor_get(t, bytes.data() + start, off, n);
    }
    size_t n_bytes() override { return bytes.size(); }
};
struct reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes; size_t pos = 0;
    explicit reader(const std::vector<uint8_t> & b) : bytes(b) {}
    const uint8_t * read(size_t n) override {
        require(n <= bytes.size() - pos); const auto * p = bytes.data() + pos; pos += n; return p;
    }
    void read_to(void * p, size_t n) override { std::memcpy(p, read(n), n); }
    size_t n_bytes() override { return pos; }
};

static void check(bool v41, int n) {
    llama_model model(llama_model_default_params());
    model.arch = v41 ? LLM_ARCH_DEEPSEEK41 : LLM_ARCH_LLAMA;
    auto & hp = model.hparams;
    hp.n_layer = 2; hp.n_embd = 4; hp.n_ctx_train = 1024;
    hp.n_head_arr.fill(1); hp.n_head_kv_arr.fill(1);
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 4;
    hp.n_embd_head_k_swa = hp.n_embd_head_v_swa = 4;
    hp.indexer_head_size = 4; hp.n_swa = 128; hp.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hp.swa_layers.fill(1); hp.dsv4_compress_ratios[0] = v41 ? 2 : 4;
    hp.dsv4_compress_ratios[1] = v41 ? 1 : 128;
    hp.dsv41_kv_source.fill(-1); hp.dsv41_index_key_source.fill(-1);
    for (int l = 0; l < 2; ++l) { hp.dsv41_kv_source[l] = l; hp.dsv41_index_key_source[l] = l; }
    llama_kv_cache_dsv4 kv(model, GGML_TYPE_F32, GGML_TYPE_F32, false, false, false, false,
        1024, 1, 512, 1, nullptr, nullptr);
    llama_vocab vocab;
    std::vector<float> embd(4*n);
    std::vector<llama_pos> pos(n); std::vector<int32_t> ns(n, 1);
    llama_seq_id seq = 0; std::vector<llama_seq_id *> ids(n + 1, &seq); ids[n] = nullptr;
    std::vector<int8_t> outputs(n, 0);
    for (int i = 0; i < n; ++i) { pos[i] = i; }
    llama_batch b = {n, nullptr, embd.data(), pos.data(), ns.data(), ids.data(), outputs.data()};
    llama_batch_allocr alloc(1);
    require(alloc.init(b, vocab, &kv, 4, 1, false));
    auto mc = kv.init_batch(alloc, 512, false);
    require(mc->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    do { require(mc->apply()); } while (mc->next());
    require(kv.seq_pos_max(0) == n - 1);

    // Row-specific sentinels: a serializer that drops live rows must fail the readback below.
    struct saved_tensor { ggml_tensor * tensor; std::vector<float> values; };
    std::vector<saved_tensor> saved;
    auto seed = [&](ggml_tensor * t, size_t count) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) { values[i] = float(i + 100*saved.size() + 1); }
        ggml_backend_tensor_set(t, values.data(), 0, values.size()*sizeof(float));
        saved.push_back({t, std::move(values)});
    };
    for (auto * cache : {kv.get_csa(), kv.get_hca(), kv.get_lid()}) {
        const unsigned ratio = cache == kv.get_hca() ? 128 : v41 ? 1 : 4;
        for (auto l : cache->get_layer_ids()) { seed(cache->get_k_storage(l), 4*(n/ratio)); }
    }
    ggml_context_ptr views(ggml_init({64*ggml_tensor_overhead(), nullptr, true}));
    for (auto * state : {kv.get_csa_state(), kv.get_hca_state(), kv.get_lid_state()}) {
        for (int l = 0; l < 2; ++l) {
            // V4.1's index keys have no compressor state; the other states own layers of their ratio.
            if ((v41 && state == kv.get_lid_state()) || hp.dsv4_compress_ratios[l] != state->get_ratio()) { continue; }
            for (auto * t : {state->get_kv_all(views.get(), l), state->get_score_all(views.get(), l)}) {
                ggml_backend_view_init(t); seed(t, ggml_nelements(t));
            }
        }
    }
    writer before; kv.state_write(before, 0, 0);
    // Suffix beyond pos_max is a no-op; fixed 4/128 trim ratios used to erase V4.1 metadata here.
    require(kv.seq_rm(0, n, -1));
    writer trimmed; kv.state_write(trimmed, 0, 0); require(trimmed.bytes == before.bytes);
    kv.clear(true);
    reader in(before.bytes); kv.state_read(in, 0, 0);
    require(in.n_bytes() == before.bytes.size()); require(kv.seq_pos_max(0) == n - 1);
    for (const auto & s : saved) {
        std::vector<float> actual(s.values.size());
        ggml_backend_tensor_get(s.tensor, actual.data(), 0, actual.size()*sizeof(float));
        require(actual == s.values);
    }
    writer after; kv.state_write(after, 0, 0); require(after.bytes == before.bytes);
    std::printf("DSv4 state roundtrip ratios=%s tokens=%d bytes=%zu OK\n", v41 ? "2/1" : "4/128", n, before.bytes.size());
}
int main() {
    ggml_backend_load_all();
    try { for (bool v41 : {false, true}) { for (int n : {1, 2, 3, 127, 128, 129, 255, 256, 257, 385}) { check(v41, n); } } }
    catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
