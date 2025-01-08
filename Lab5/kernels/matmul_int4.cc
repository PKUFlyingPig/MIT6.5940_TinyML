#include <cstdlib>
#include <iostream>

#include "matmul.h"

namespace matmul {

void MatmulOperator::naive_mat_mul_int4(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *B_sc = params->scales;
    float *B_zp = params->zero_point;
    float *data_C = C->data_ptr;
    CHECK_MATRICES_int4weight(A, B, C);
#ifdef QM_ARM
    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float acc = 0;
            for (k = 0; k < A->column; k += block_size) {
                float s = B_sc[(j * A->column + k) / block_size];
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->column + k / 2];
                float *x_ptr = &A->data_ptr[i * A->column + k];
                for (int qi = 0; qi < block_size / 2; qi += 16) {
                    // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                    // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                    // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                    // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                    for (int qj = 0; qj < 16; qj++) {
                        uint8_t packed_int4_0 = weight_32_int4[qi + qj];
                        float deq_0 = (float)((packed_int4_0 & 0x0F) - 8.0) * s;
                        float deq_1 = (float)((packed_int4_0 >> 4) - 8.0) * s;
                        acc += *x_ptr * deq_0;
                        acc += x_ptr[16] * deq_1;
                        x_ptr++;
                    }
                }
            }
            C->data_ptr[i * C->column + j] = acc;
        }
    }
#else
#ifdef QM_x86
    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float acc = 0;
            for (k = 0; k < A->column; k += block_size * 2) {
                float s = B_sc[(j * A->column + k) / block_size];
                float s1 = B_sc[(j * A->column + k) / block_size + 1];
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->column + k / 2];
                float *x_ptr = &A->data_ptr[i * A->column + k];
                for (int qi = 0; qi < block_size / 2; qi += 32) {
                    // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 256 bit
                    // expected layout of inB: (0, 32), (1, 33), (2, 34), (3, 35)...
                    // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                    // high: (32, 0), (33, 0), (34, 0), (35, 0) ...
                    for (int qj = 0; qj < 32; qj++) {
                        uint8_t packed_int4_0 = weight_32_int4[qi + qj];
                        float deq_0 = (float)((packed_int4_0 & 0x0F) - 8.0) * s;
                        float deq_1 = (float)((packed_int4_0 >> 4) - 8.0) * s1;
                        acc += *x_ptr * deq_0;
                        acc += x_ptr[32] * deq_1;
                        x_ptr++;
                    }
                }
            }
            C->data_ptr[i * C->column + j] = acc;
        }
    }
#else
    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float acc = 0;

            for (k = 0; k < A->column; k += block_size) {
                float s = B_sc[(j * B->column * 2 + k) / block_size];
                float z = *B_zp;
                float *data_A = &A->data_ptr[i * A->column + k];
                uint8_t *data_B = &B->int4_data_ptr[j * B->column + k / 2];

                for (int qi = 0; qi < block_size / 2; qi++) {
                    uint8_t packed_int4_0 = data_B[qi];
                    float deq_0 = ((float)(packed_int4_0 & 0x0F) - z) * s;
                    float deq_1 = ((float)(packed_int4_0 >> 4) - z) * s;
                    acc += *data_A++ * deq_0;
                    acc += *data_A++ * deq_1;
                }
            }

            data_C[i * C->column + j] = acc;
        }
    }
#endif  // QM_x86
#endif  // QM_ARM
}

void MatmulOperator::naive_mat_mul_int4_with_offset(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *B_sc = params->scales;
    float *B_offset = params->offset;
    float *B_zp = params->zero_point;
    float *data_C = C->data_ptr;
    CHECK_MATRICES_int4weight(A, B, C);

    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float acc = 0;

            for (k = 0; k < A->column; k += block_size) {
                float s = B_sc[(j * B->column * 2 + k) / block_size];
                float o = B_offset[(j * B->column * 2 + k) / block_size];
                float z = *B_zp;
                float *data_A = &A->data_ptr[i * A->column + k];
                uint8_t *data_B = &B->int4_data_ptr[j * B->column + k / 2];

                for (int qi = 0; qi < block_size / 2; qi++) {
                    uint8_t packed_int4_0 = data_B[qi];
                    float deq_0 = ((float)(packed_int4_0 & 0x0F) - z) * s + o;
                    float deq_1 = ((float)(packed_int4_0 >> 4) - z) * s + o;
                    acc += *data_A++ * deq_0;
                    acc += *data_A++ * deq_1;
                }
            }

            data_C[i * C->column + j] = acc;
        }
    }
}
}  // namespace matmul
