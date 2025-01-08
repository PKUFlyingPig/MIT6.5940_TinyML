#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"
struct multithreading_loop_unrolling_thread_args {
    int start, end;
    const struct matmul_params *params;
};
static void *multithreading_loop_unrolling_worker_func(void *args) {
    struct multithreading_loop_unrolling_thread_args *mat_args =
        (struct multithreading_loop_unrolling_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start; col < mat_args->end; col += 4) {
            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int8 activation
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];
                // pointer of the int4 weights
                uint8_t *w0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                uint8_t *w1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];
                uint8_t *w2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];
                uint8_t *w3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];
                // scale of weight
                float s_w0 = params->scales[(col * k + ch) / block_size];
                float s_w1 = params->scales[((col + 1) * k + ch) / block_size];
                float s_w2 = params->scales[((col + 2) * k + ch) / block_size];
                float s_w3 = params->scales[((col + 3) * k + ch) / block_size];
#ifdef QM_ARM
                // order of weights with QM_ARM:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)
                // QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         128 bit                         127
                // process 16 bytes of weigths (128 bit) = 1 block for each of unrolled `col`
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                for (int qj = 0; qj < 16; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    signed char w0_low = (w0_int4[qj] & 0x0F) - 8.0;
                    signed char w0_high = (w0_int4[qj] >> 4) - 8.0;
                    signed char w1_low = (w1_int4[qj] & 0x0F) - 8.0;
                    signed char w1_high = (w1_int4[qj] >> 4) - 8.0;
                    signed char w2_low = (w2_int4[qj] & 0x0F) - 8.0;
                    signed char w2_high = (w2_int4[qj] >> 4) - 8.0;
                    signed char w3_low = (w3_int4[qj] & 0x0F) - 8.0;
                    signed char w3_high = (w3_int4[qj] >> 4) - 8.0;

                    // multiply and accumulate
                    intermediate_sum0 += a_int8[qj] * w0_low + a_int8[qj + 16] * w0_high;
                    intermediate_sum1 += a_int8[qj] * w1_low + a_int8[qj + 16] * w1_high;
                    intermediate_sum2 += a_int8[qj] * w2_low + a_int8[qj + 16] * w2_high;
                    intermediate_sum3 += a_int8[qj] * w3_low + a_int8[qj + 16] * w3_high;
                }
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                ch += block_size;
#endif
#ifdef QM_x86
                // scales of the second block
                float s_w0_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_w1_2nd = params->scales[((col + 1) * k + ch) / block_size + 1];
                float s_w2_2nd = params->scales[((col + 2) * k + ch) / block_size + 1];
                float s_w3_2nd = params->scales[((col + 3) * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];
                // order of weights with QM_x86:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
                // QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         256 bit
                // process 32 bytes of weigths (256 bit) = 2 blocks for each of unrolled `col`
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                int intermediate_sum0_2nd = 0, intermediate_sum1_2nd = 0, intermediate_sum2_2nd = 0,
                    intermediate_sum3_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {
                    // TODO: decode a packed byte into two int8 in the range of (-8, 7)

                    // TODO: int8 multiply and accumulate operation
                }
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc0 += (float)intermediate_sum0_2nd * s_a_2nd * s_w0_2nd;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc1 += (float)intermediate_sum1_2nd * s_a_2nd * s_w1_2nd;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc2 += (float)intermediate_sum2_2nd * s_a_2nd * s_w2_2nd;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                acc3 += (float)intermediate_sum3_2nd * s_a_2nd * s_w3_2nd;
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
            C->data_ptr[row * n + col + 2] = acc2;
            C->data_ptr[row * n + col + 3] = acc3;
        }
    }
    return NULL;
}

namespace matmul {
void MatmulOperator::mat_mul_multithreading_loop_unrolling(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    assert(params->block_size % 32 == 0);  // support block size to be multiples of 32
    assert(A->row == C->row);              // support block size to be multiples of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;

    const int num_thread = 4;
    pthread_t thread_pool[num_thread];
    struct multithreading_loop_unrolling_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    // Thread creation
    int cols_per_thread = (n + num_thread - 1) / num_thread;
    for (int i = 0; i < num_thread; i++) {
        threads_args[i].start = i * cols_per_thread;
        threads_args[i].end = std::min((i + 1) * cols_per_thread, n);
        threads_args[i].params = params;
        pthread_create(&thread_pool[i], NULL, multithreading_loop_unrolling_worker_func, &threads_args[i]);
    }

    // Join threads
    for (int i = 0; i < num_thread; i++) {
        pthread_join(thread_pool[i], NULL);
    }
};
}  // namespace matmul
