// SPDX-License-Identifier: MIT
#include "llama-dspark-quant.h"
#include <cstdio>
#include <stdexcept>

static void require(bool ok) { if (!ok) { throw std::runtime_error("DSpark FP8 quantization mismatch"); } }
int main() {
    try {
        float input[32] = {}, output[32];
        input[31] = 448.0f; // scale exactly 1, irrespective of the other values
        for (unsigned code = 0; code < 127; ++code) {
            const int e = code >> 3, m = code & 7;
            const float value = e ? std::ldexp(float(8+m),e-10) : std::ldexp(float(m),-9);
            for (float sign : {1.0f,-1.0f}) {
                input[0] = sign*value;
                llama_dspark_act_quant32(input,output);
                require(output[0] == input[0]);
                require(std::signbit(output[0]) == std::signbit(input[0]));
            }
        }
        input[0] = 1.0625f; input[1] = 1.1875f;
        input[2] = std::ldexp(1.0f,-10); input[3] = std::ldexp(3.0f,-10);
        llama_dspark_act_quant32(input,output);
        require(output[0] == 1.0f && output[1] == 1.25f);
        require(output[2] == 0.0f && output[3] == std::ldexp(1.0f,-8));
        for (float & v : input) { v = 0.0f; }
        llama_dspark_act_quant32(input,output);
        for (float v : output) { require(v == 0.0f); }
        for (float & v : input) { v = 450.0f; }
        llama_dspark_act_quant32(input,output);
        for (float v : output) { require(v == 448.0f); } // ceil scale=2, nearest E4M3=224
        std::puts("DSpark FP8/E8M0/BF16: 254 code roundtrips, ties, zero and scale boundary OK");
    } catch (const std::exception & e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
