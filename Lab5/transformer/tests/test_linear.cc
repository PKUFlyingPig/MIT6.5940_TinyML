#include <cmath>

#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_FPLinear_int4() {
    const int m = 1, n = 32000, k = 4096;

    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(m * k), 1, m, k);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(n * k), 1, n, k);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> output(mem_buf.get_fpbuffer(m * n), 1, m, n);

    hidden_states.load("tests/assets/input.bin");
    outputGT.load("tests/assets/output.bin");

    // quantize the weight to int4
    Matrix3D<uint8_t> int4_weight((uint8_t *)mem_buf.get_int8buffer(n * k / 2), 1, n, k / 2);
    // Linear_FP_int4 int4_op;
    Linear_FP_int4 int4_op = Linear_FP_int4(int4_weight, "INT4/models/LLaMA_7B_2_chat/lm_head/");

    Matrix3D<float> outputQ(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> outputQ_simd(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> outputQ_fast(mem_buf.get_fpbuffer(m * n), 1, m, n);

    // warm up
    for (int i = 0; i < 1; i++) {
        int4_op.forward(hidden_states, outputQ_fast);
    }

    const int flops = k * m * n * 2;
    int4_op.forward_ref(hidden_states, outputQ);

    for (int i = 0; i < 10; i++) {
        STATS_FLOPS(int4_op.profile_name, flops);
        int4_op.forward(hidden_states, outputQ_fast);
        STATS_END(int4_op.profile_name);
    }
    bool success = check_two_equal(outputQ.m_data, outputQ_fast.m_data, outputQ_fast.length(), 1e-3);

    if (!success) {
        std::cout << "-------- Sanity check of " << int4_op.profile_name << " implementation: Fail! -------- "
                  << std::endl;
        exit(-1);
    } else
        std::cout << "-------- Sanity check of " << int4_op.profile_name << " implementation: Passed! -------- "
                  << std::endl;
}

int main() {
    test_FPLinear_int4();
    Profiler::getInstance().report_internal();
}
