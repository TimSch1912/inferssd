// SPDX-License-Identifier: MIT
#pragma once
#include "llama-model.h"
#include "ggml-cpp.h"

// Owned separately: main graph and KV still have exactly 40 layers.
struct llama_dspark_model {
    gguf_context_ptr file;
    ggml_context_ptr metadata;
    ggml_context_ptr dense;
    ggml_backend_buffer_ptr dense_buffer;
    ggml_backend_buffer_ptr expert_buffer;
    std::array<llama_layer, 3> layers;
    ggml_tensor * main_proj = nullptr;
    ggml_tensor * main_norm = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * markov_embd = nullptr;
    ggml_tensor * markov_output = nullptr;
    ggml_tensor * confidence = nullptr;
    std::string index_path;
};

struct llama_dspark_input {
    const float * features;
    const llama_pos * positions;
    int32_t n_context;
    llama_token known;
    int32_t n_drafts;
};
