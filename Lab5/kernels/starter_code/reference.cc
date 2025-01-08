#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

namespace matmul {
void MatmulOperator::mat_mul_reference(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            float acc = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int4 weights
                uint8_t *w_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                // pointer of the int8 activation
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];
                // scale of weight
                float s_w = params->scales[(col * k + ch) / block_size];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];
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
                // process 16 bytes of weigths (128 bit) = 1 block
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum = 0;
                for (int qj = 0; qj < 16; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w_int4[qj];
                    signed char w_de_0 = (packed_int4_0 & 0x0F) - 8.0;
                    signed char w_de_16 = (packed_int4_0 >> 4) - 8.0;
                    // int8 multiply and accumulate operation
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum += a_int8[qj + 16] * w_de_16;
                }
                // dequantize the sum into floating point
                acc += (float)intermediate_sum * s_a * s_w;
                ch += block_size;
#endif
#ifdef QM_x86
                // scales of the second block
                float s_w_2nd = params->scales[(col * k + ch) / block_size + 1];
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
                // process 32 bytes of weigths (256 bit) = 2 blocks
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum = 0, intermediate_sum_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w_int4[qj];
                    signed char w_de_0 = (packed_int4_0 & 0x0F) - 8.0;
                    signed char w_de_16 = (packed_int4_0 >> 4) - 8.0;
                    // int8 multiply and accumulate operation
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum_2nd += a_int8[qj + 32] * w_de_16;
                }
                // dequantize the sum into floating point
                acc += (float)intermediate_sum * s_a * s_w;
                acc += (float)intermediate_sum_2nd * s_a_2nd * s_w_2nd;
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc;
        }
    }
};
}  // namespace matmul
