// SPDX-License-Identifier: MIT
#include "llama-dspark.h"
#include "llama-impl.h"
#include "gguf.h"
#include "../vendor/nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

bool llama_model_has_mtp_sidecar(const llama_model * model) { return model && model->dspark != nullptr; }

bool llama_model_load_mtp_sidecar(llama_model * model, const char * path) {
    try {
        if (!model || !path || model->dspark || model->arch != LLM_ARCH_DEEPSEEK41 ||
            model->hparams.n_layer != 40 || model->hparams.n_embd != 5120 || model->vocab.n_tokens() != 129280 ||
            model->hparams.dsv4_hc_mult != 4 || model->hparams.n_expert != 384) {
            throw std::runtime_error("DSpark requires the DeepSeek-V4.1-Flash-1017 main model, loaded once before contexts");
        }
        auto mtp = std::make_unique<llama_dspark_model>();
        ggml_context * meta = nullptr;
        mtp->file.reset(gguf_init_from_file(path, {true, &meta}));
        mtp->metadata.reset(meta);
        if (!mtp->file || !meta) { throw std::runtime_error("cannot read MTP GGUF"); }
        auto * uf = mtp->file.get();
        auto str = [&](const char * key) {
            const auto id = gguf_find_key(uf, key);
            if (id < 0 || gguf_get_kv_type(uf, id) != GGUF_TYPE_STRING) { throw std::runtime_error(key); }
            return std::string(gguf_get_val_str(uf, id));
        };
        if (str("general.architecture") != "deepseek41" || str("general.type") != "mtp-sidecar" ||
            str("deepseek41.mtp.format") != "inferssd-dspark-v1" || gguf_get_n_tensors(uf) != 81) {
            throw std::runtime_error("unsupported DSpark sidecar format");
        }
        const auto cfg = nlohmann::json::parse(str("deepseek41.mtp.config")).at("text_config");
        const nlohmann::json expected = {
            {"hidden_size",5120}, {"vocab_size",129280}, {"num_hidden_layers",40},
            {"hc_mult",4}, {"num_nextn_predict_layers",3}, {"dspark_block_size",5},
            {"dspark_noise_token_id",128799}, {"dspark_markov_rank",256},
            {"dspark_n_routed_experts",128}, {"dspark_num_experts_per_tok",3},
            {"n_shared_experts",1}, {"swiglu_limit",10.0}, {"rms_norm_eps",1e-20},
            {"routed_scaling_factor",1.5}, {"sliding_window",128},
            {"dspark_target_layer_ids",{37,38,39}}};
        for (const auto & entry : expected.items()) {
            if (cfg.at(entry.key()) != entry.value()) { throw std::runtime_error("MTP config mismatch: " + entry.key()); }
        }
        const uint64_t file_size = std::filesystem::file_size(path);
        for (int64_t i = 0; i < gguf_get_n_tensors(uf); ++i) {
            const uint64_t offset = gguf_get_data_offset(uf) + gguf_get_tensor_offset(uf, i);
            const auto * t = ggml_get_tensor(meta, gguf_get_tensor_name(uf, i));
            if (offset > file_size || ggml_nbytes(t) > file_size - offset) { throw std::runtime_error("truncated MTP tensor"); }
        }
        mtp->index_path = (std::filesystem::path(path).parent_path() / "mtp-experts.tsv").string();
        if (!std::filesystem::is_regular_file(mtp->index_path)) { throw std::runtime_error("missing mtp-experts.tsv"); }
        mtp->dense.reset(ggml_init({100*ggml_tensor_overhead(), nullptr, true}));
        if (!mtp->dense) { throw std::runtime_error("cannot allocate MTP metadata"); }
        std::set<std::string> used;
        auto tensor = [&](const std::string & name, ggml_type type, std::vector<int64_t> shape, bool expert = false) {
            auto * src = ggml_get_tensor(meta, name.c_str());
            if (!src || src->type != type) { throw std::runtime_error("MTP missing/wrong tensor type: " + name); }
            shape.resize(4, 1);
            for (int d = 0; d < 4; ++d) {
                if (src->ne[d] != shape[d]) { throw std::runtime_error("MTP tensor shape mismatch: " + name); }
            }
            used.insert(name);
            if (expert) { return src; } // Metadata only: payload is owned by the shared SSD source index.
            auto * dst = ggml_dup_tensor(mtp->dense.get(), src);
            ggml_set_name(dst, name.c_str()); return dst;
        };
        for (int k = 0; k < 3; ++k) {
            auto & l = mtp->layers[k];
            const std::vector<std::pair<const char *, ggml_tensor **>> fields = {
                {"attn_norm.weight",&l.attn_norm}, {"attn_sinks.weight",&l.attn_sinks},
                {"attn_q_a.weight",&l.wq_a}, {"attn_q_a_norm.weight",&l.attn_q_a_norm},
                {"attn_q_b.weight",&l.wq_b}, {"attn_kv.weight",&l.wkv},
                {"attn_kv_a_norm.weight",&l.attn_kv_norm}, {"attn_output_a.weight",&l.wo_a},
                {"attn_output_b.weight",&l.wo_b}, {"hc_attn_fn.weight",&l.hc_attn_fn},
                {"hc_attn_base.weight",&l.hc_attn_base}, {"hc_attn_scale.weight",&l.hc_attn_scale},
                {"hc_ffn_fn.weight",&l.hc_ffn_fn}, {"hc_ffn_base.weight",&l.hc_ffn_base},
                {"hc_ffn_scale.weight",&l.hc_ffn_scale}, {"ffn_gate_inp.weight",&l.ffn_gate_inp},
                {"exp_probs_b.bias",&l.ffn_exp_probs_b}, {"exp_probs_b_vl.bias",&l.ffn_exp_probs_b_vl},
                {"ffn_norm.weight",&l.ffn_norm}, {"ffn_gate_exps.weight",&l.ffn_gate_exps},
                {"ffn_up_exps.weight",&l.ffn_up_exps}, {"ffn_down_exps.weight",&l.ffn_down_exps},
                {"ffn_gate_shexp.weight",&l.ffn_gate_shexp}, {"ffn_up_shexp.weight",&l.ffn_up_shexp},
                {"ffn_down_shexp.weight",&l.ffn_down_shexp}};
            for (const auto & f : fields) {
                const auto * main = model->get_tensor(("blk.37." + std::string(f.first)).c_str());
                if (!main) { throw std::runtime_error("missing main tensor analogue: " + std::string(f.first)); }
                std::vector<int64_t> shape(main->ne, main->ne + 4);
                for (auto & n : shape) { if (n == 384) { n = 128; } }
                *f.second = tensor("blk." + std::to_string(40+k) + "." + f.first, main->type, shape,
                    std::string(f.first).find("_exps.") != std::string::npos);
            }
        }
        mtp->main_proj = tensor("blk.40.nextn.main_proj.weight", GGML_TYPE_Q8_0, {15360,5120});
        mtp->main_norm = tensor("blk.40.nextn.main_norm.weight", GGML_TYPE_F32, {5120});
        mtp->norm = tensor("blk.42.nextn.shared_head_norm.weight", GGML_TYPE_F32, {5120});
        mtp->markov_embd = tensor("blk.42.nextn.markov_embd.weight", GGML_TYPE_BF16, {256,129280});
        mtp->markov_output = tensor("blk.42.nextn.markov_output.weight", GGML_TYPE_BF16, {256,129280});
        mtp->confidence = tensor("blk.42.nextn.confidence_proj.weight", GGML_TYPE_BF16, {5376,1});
        if (used.size() != 81) { throw std::runtime_error("unrecognized MTP tensor"); }
        const auto buft = ggml_backend_dev_buffer_type(model->dev_layer(39));
        mtp->dense_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(mtp->dense.get(), buft));
        mtp->expert_buffer.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 0));
        if (!mtp->dense_buffer || !mtp->expert_buffer) { throw std::runtime_error("cannot allocate MTP weights"); }
        ggml_backend_buffer_set_usage(mtp->dense_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & l : mtp->layers) {
            for (auto * t : {l.ffn_up_exps,l.ffn_gate_exps,l.ffn_down_exps}) { t->buffer = mtp->expert_buffer.get(); }
        }
        std::ifstream input(path, std::ios::binary);
        std::vector<char> chunk(16*1024*1024);
        for (auto * t = ggml_get_first_tensor(mtp->dense.get()); t; t = ggml_get_next_tensor(mtp->dense.get(), t)) {
            const auto i = gguf_find_tensor(uf, t->name);
            input.seekg(gguf_get_data_offset(uf) + gguf_get_tensor_offset(uf,i));
            for (size_t off = 0; off < ggml_nbytes(t);) {
                const auto n = std::min(chunk.size(), ggml_nbytes(t)-off);
                if (!input.read(chunk.data(), n)) { throw std::runtime_error("failed reading MTP weights"); }
                ggml_backend_tensor_set(t, chunk.data(), off, n); off += n;
            }
        }
        // These array entries describe only the draft experts; n_layer remains 40.
        for (int l = 40; l < 43; ++l) {
            model->hparams.swiglu_clamp_exp[l] = 10.0f;
            model->hparams.swiglu_clamp_shexp[l] = 10.0f;
            model->hparams.dsv4_compress_ratios[l] = 0;
        }
        model->dspark = std::move(mtp);
        LLAMA_LOG_INFO("DSpark: loaded 3 stages, 128 experts/stage, main graph/KV stays at 40 layers\n");
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("DSpark sidecar: %s\n",e.what()); return false;
    }
}
