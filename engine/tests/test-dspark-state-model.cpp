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
#include <cstdlib>
#include <map>
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
            if (std::getenv("DSPARK_UNDO_CANARY")) {
                auto restore = [&] { require(llama_state_seq_set_data_ext(ctx,state.data(),state.size(),0,flags)==state.size(),"restore failed"); };
                std::vector<llama_token> inputs = {known[0]};
                for (int i = 1; i < 5; ++i) { inputs.push_back(text[(n+i)%text.size()]); }
                std::map<int,std::pair<std::vector<float>,std::vector<float>>> references;
                auto difference = [&](const std::vector<float> & x, const std::vector<float> & y) {
                    double mean=0; float maximum=0;
                    for (int i=0;i<width;++i) {
                        require(std::isfinite(x[i]) && std::isfinite(y[i]),"nonfinite canary logit");
                        const auto d=std::abs(x[i]-y[i]);mean+=d;maximum=std::max(maximum,d);
                    }
                    return std::make_pair(mean/width,maximum);
                };
                // k=1..4, every retained prefix. The independent full-restore control
                // touches the same future experts, separating tier arithmetic from KV damage.
                for (int batch_size=2;batch_size<=5;++batch_size) {
                    std::vector<llama_token> verifier(inputs.begin(),inputs.begin()+batch_size);
                    for (int keep=1;keep<=batch_size;++keep) {
                        std::vector<llama_token> probe{text[(n+7)%text.size()]};
                        auto replay = [&] {
                            for (int i=0;i<keep;++i) { std::vector<llama_token> one{inputs[i]};decode(one,n+i); }
                            decode(probe,n+keep);return logits();
                        };
                        auto sequential = [&] { restore();return replay(); };
                        auto & saved=references[keep];
                        if (saved.first.empty()) { saved.first=sequential();saved.second=sequential(); }
                        const auto clean=difference(saved.first,saved.second);
                        restore();decode(verifier,n);restore();
                        const auto dirty_control=replay();
                        const auto dirty=difference(saved.first,dirty_control);
                        const double control_mean=std::max(clean.first,dirty.first);
                        const float control_max=std::max(clean.second,dirty.second);
                        const double mean_limit=std::max(0.02,3*control_mean);
                        const float max_limit=std::max(0.25f,3*control_max);
                        std::vector<float> full_reference;
                        std::pair<double,float> full_variance{0,0};
                        if (keep==batch_size) {
                            // Independent fresh batches span cold-to-warm expert admission before
                            // testing Full Accept. They do not use the bounded-undo implementation.
                            for (int repeat=0;repeat<4;++repeat) {
                                restore();decode(verifier,n);decode(probe,n+keep);auto fresh=logits();
                                if (full_reference.empty()) { full_reference=std::move(fresh); }
                                else {
                                    const auto d=difference(full_reference,fresh);
                                    full_variance.first=std::max(full_variance.first,d.first);
                                    full_variance.second=std::max(full_variance.second,d.second);
                                }
                            }
                        }
                        auto snapshot = [&] {
                            std::vector<uint8_t> bytes(llama_state_seq_get_size_ext(ctx,0,flags));
                            require(llama_state_seq_get_data_ext(ctx,bytes.data(),bytes.size(),0,flags)==bytes.size(),"cache snapshot failed");
                            return bytes;
                        };
                        restore();
                        require(llama_dspark_undo_begin(ctx),"undo begin failed");
                        decode(verifier,n);
                        const auto full_before=keep==batch_size ? snapshot() : std::vector<uint8_t>{};
                        require(llama_dspark_undo_end(ctx,keep!=batch_size),"undo end failed");
                        if (keep==batch_size) {
                            require(snapshot()==full_before,"Full Accept mutated cache on discard");
                            printf("MODEL_FULL_BYTES tokens=%d batch=%d exact=1 bytes=%zu\n",n,batch_size,full_before.size());fflush(stdout);
                        }
                        if (keep!=batch_size) {
                            // Test-only full serialization: production Fast never snapshots the sequence.
                            std::vector<uint8_t> restored(state.size());
                            require(llama_state_seq_get_data_ext(ctx,restored.data(),restored.size(),0,flags)==state.size(),"undo state size changed");
                            require(restored==state,"delta undo did not restore exact cache bytes");
                            printf("MODEL_CACHE_BYTES tokens=%d batch=%d keep=%d exact=1 bytes=%zu\n",n,batch_size,keep,state.size());fflush(stdout);
                        }
                        const auto actual=keep!=batch_size ? replay() : (decode(probe,n+keep),logits());
                        const auto gap=difference(saved.first,actual);
                        printf("MODEL_UNDO tokens=%d batch=%d keep=%d max_delta=%.9g mean_delta=%.9g control_max=%.9g control_mean=%.9g mean_limit=%.9g max_limit=%.9g\n",
                            n,batch_size,keep,gap.second,gap.first,control_max,control_mean,mean_limit,max_limit);fflush(stdout);
                        if (keep!=batch_size) {
                            require(gap.first<=mean_limit && gap.second<=max_limit,"undo continuation outside measured control envelope");
                        } else {
                            const auto actual_gap=difference(actual,full_reference);
                            const double full_mean_limit=std::max(0.02,3*full_variance.first);
                            const float full_max_limit=std::max(0.25f,3*full_variance.second);
                            printf("MODEL_FULL_ACCEPT tokens=%d batch=%d fresh_max=%.9g fresh_mean=%.9g control_max=%.9g control_mean=%.9g mean_limit=%.9g max_limit=%.9g\n",
                                n,batch_size,actual_gap.second,actual_gap.first,full_variance.second,full_variance.first,full_mean_limit,full_max_limit);fflush(stdout);
                            require(actual_gap.first<=full_mean_limit && actual_gap.second<=full_max_limit,"full accept differs beyond measured control envelope");
                        }
                    }
                }
                continue;
            }
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
