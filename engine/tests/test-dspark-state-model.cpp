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
                // k=1..4, every retained prefix, including full acceptance.
                for (int batch_size = 2; batch_size <= 5; ++batch_size) {
                for (int keep = 1; keep <= batch_size; ++keep) {
                    auto sequential = [&] {
                        restore();
                        for (int i = 0; i < keep; ++i) { std::vector<llama_token> one{inputs[i]}; decode(one,n+i); }
                        std::vector<llama_token> probe{text[(n+7)%text.size()]}; decode(probe,n+keep);
                        return logits();
                    };
                    auto & saved = references[keep];
                    if (saved.first.empty()) { saved.first=sequential();saved.second=sequential(); }
                    const auto & reference=saved.first; const auto & control=saved.second;
                    restore();
                    require(llama_dspark_undo_begin(ctx),"undo begin failed");
                    std::vector<llama_token> verifier(inputs.begin(),inputs.begin()+batch_size);
                    decode(verifier,n);
                    require(llama_dspark_undo_end(ctx,keep != batch_size),"undo end failed");
                    if (keep != batch_size) {
                        for (int i=0;i<keep;++i) { std::vector<llama_token> one{inputs[i]}; decode(one,n+i); }
                    }
                    std::vector<llama_token> probe{text[(n+7)%text.size()]}; decode(probe,n+keep);
                    const auto actual = logits();
                    double mean=0, control_mean=0; float maximum=0, control_max=0;
                    for (int i=0;i<width;++i) {
                        const auto d=std::abs(reference[i]-actual[i]), c=std::abs(reference[i]-control[i]);
                        require(std::isfinite(actual[i]),"nonfinite canary logit");
                        mean+=d; control_mean+=c; maximum=std::max(maximum,d); control_max=std::max(control_max,c);
                    }
                    mean/=width; control_mean/=width;
                    std::printf("MODEL_UNDO tokens=%d batch=%d keep=%d max_delta=%.9g mean_delta=%.9g control_max=%.9g control_mean=%.9g\n",
                        n,batch_size,keep,maximum,mean,control_max,control_mean); std::fflush(stdout);
                    // Mixed-tier placement changes arithmetic. Bound both average drift and outliers;
                    // retain the independently repeated sequential control alongside every comparison.
                    if (keep != batch_size) {
                        require(mean < 0.02 && maximum < 0.25,"undo continuation outside normal numerical tolerance");
                    } else {
                        // No undo on full acceptance. Isolate ordinary multi-token arithmetic from undo:
                        // compare the committed batch with a fresh batch using exactly the same inputs.
                        restore(); decode(verifier,n); decode(probe,n+keep); const auto fresh=logits();
                        double gap=0; float worst=0;
                        for (int i=0;i<width;++i) { const auto d=std::abs(fresh[i]-actual[i]);gap+=d;worst=std::max(worst,d); }
                        gap/=width;
                        printf("MODEL_FULL_ACCEPT tokens=%d batch=%d fresh_max=%.9g fresh_mean=%.9g\n",n,batch_size,worst,gap);fflush(stdout);
                        require(gap < 0.02 && worst < 0.25,"full-accept state differs from fresh batch");
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
