// SPDX-License-Identifier: MIT
#pragma once
#include "llama-engram.h"
#include <algorithm>
#include <array>
#include <cmath>

// model.py/kernel.py: 32 activations, E4M3FN, ceil-power-of-two E8M0 scale,
// fused quant/dequant back to BF16. Used only by the opt-in draft graph.
inline void llama_dspark_act_quant32(const float * input, float * output) {
    static const auto levels = [] {
        std::array<float,127> values{};
        for (unsigned i = 0; i < values.size(); ++i) {
            const int exponent = i >> 3, mantissa = i & 7;
            values[i] = exponent ? std::ldexp(float(8+mantissa),exponent-10) : std::ldexp(float(mantissa),-9);
        }
        return values;
    }();
    float values[32], amax = 1e-4f;
    for (int i = 0; i < 32; ++i) {
        values[i] = llama_dsv41_engram_bf16_round(input[i]);
        amax = std::max(amax,std::abs(values[i]));
    }
    int exponent;
    const float mantissa = std::frexp(amax*(1.0f/448.0f),&exponent);
    const float scale = std::ldexp(1.0f, exponent-(mantissa == 0.5f));
    for (int i = 0; i < 32; ++i) {
        const float x = std::min(448.0f,std::abs(values[i]/scale));
        auto hi = std::lower_bound(levels.begin(),levels.end(),x);
        size_t code = hi-levels.begin();
        if (code > 0) {
            const float up = levels[code]-x, down = x-levels[code-1];
            if (down < up || (down == up && code%2)) { --code; }
        }
        output[i] = llama_dsv41_engram_bf16_round(std::copysign(levels[code]*scale,values[i]));
    }
}
