// SPDX-License-Identifier: MIT
// Real-model full-state rollback canary. Run with fixed expert placement (e.g. --ssd-vram-budget 0).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ssd.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }
int main(int argc, char ** argv) {
    common_params params;
    params.n_parallel = 1; params.n_ctx = 8192; params.n_batch = params.n_ubatch = 8192;
    common_init();
    if (!common_params_parse(argc,argv,params,LLAMA_EXAMPLE_SERVER)) { return 1; }
    try {
        auto init = common_init_from_params(params);
        auto * ctx = init->context(); auto * model = init->model();
        require(ctx && model,"model/context initialization failed");
        const int width = llama_vocab_n_tokens(llama_model_get_vocab(model));
        auto text = common_tokenize(ctx,"The quick brown fox jumps over the lazy dog. ",true);
        auto logits = [&] {
            llama_synchronize(ctx);
            const auto * p = llama_get_logits_ith(ctx,-1);
            require(p,"missing logits"); return std::vector<float>(p,p+width);
        };
        auto decode = [&](std::vector<llama_token> & tokens, int pos) {
            auto batch = llama_batch_init(tokens.size(),0,1);
            for (size_t i=0;i<tokens.size();++i) { common_batch_add(batch,tokens[i],pos+i,{0},i+1==tokens.size()); }
            const int rc = llama_decode(ctx,batch); llama_batch_free(batch); require(rc==0,"decode failed");
        };
        constexpr auto flags = LLAMA_STATE_SEQ_FLAGS_EXACT_LAYOUT;
        for (int n : {127,128,129,255,256,257,4104}) {
            llama_memory_clear(llama_get_memory(ctx),true);
            require(llama_ssd_prefill_phase(ctx,true,n,n),"prefill transition failed");
            std::vector<llama_token> prompt;
            for (int i=0;i<n;++i) { prompt.push_back(text[i%text.size()]); }
            decode(prompt,0);
            require(llama_ssd_prefill_phase(ctx,false,n,n),"decode transition failed");
            auto prefix_logits = logits();
            std::vector<llama_token> known = {llama_token(std::max_element(prefix_logits.begin(),prefix_logits.end())-prefix_logits.begin())};
            std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx,0,flags));
            require(llama_state_seq_get_data_ext(ctx,state.data(),state.size(),0,flags)==state.size(),"snapshot failed");
            decode(known,n); const auto reference = logits();
            auto restore = [&] { require(llama_state_seq_set_data_ext(ctx,state.data(),state.size(),0,flags)==state.size(),"restore failed"); };
            for (bool dirty : {false,true}) {
                restore();
                if (dirty) { std::vector<llama_token> verifier = {known[0],llama_token((known[0]+1)%width)}; decode(verifier,n); restore(); }
                decode(known,n); const auto actual = logits();
                float delta=0;
                for (int i=0;i<width;++i) { delta=std::max(delta,std::abs(reference[i]-actual[i])); }
                const bool exact=std::memcmp(reference.data(),actual.data(),width*sizeof(float))==0;
                std::printf("MODEL_STATE tokens=%d dirty=%d bitwise=%d max_delta=%.9g bytes=%zu\n",n,dirty,exact,delta,state.size());
                std::fflush(stdout); require(exact,"model logits changed after full-state rollback");
            }
        }
    } catch (const std::exception & e) { std::fprintf(stderr,"MODEL_STATE_FAIL: %s\n",e.what()); return 1; }
}
