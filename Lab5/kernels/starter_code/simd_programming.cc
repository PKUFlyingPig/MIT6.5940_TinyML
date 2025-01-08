#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif
namespace matmul {
void MatmulOperator::mat_mul_simd_programming(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
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
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            // pointer of the int4 weights
            const unsigned char *w_start = &B->int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const signed char *a_start = &A->int8_data_ptr[row * k];
            // scale of activation
            float *s_a = &params->A_scales[row * k / 32];
            // scale of weight
            float *s_w = &params->scales[col * k / 32];

            const int num_block = k / block_size;
            // Compute each block
            for (int q = 0; q < num_block; q++) {
                // load 32x4bit (16 bytes) weight
                const uint8x16_t w0 = vld1q_u8(w_start);
                w_start += 16;

                /*
                   We will accelerate the program using ARM Intrinsics. You can check the documentation of operations
                   at: https://developer.arm.com/architectures/instruction-sets/intrinsics
                */
                // TODO: decode the lower and upper half of the weights as int8x16_t
                // Hint:
                // (1) use `vandq_u8` with the mask_low4bit to get the lower half
                // (2) use `vshrq_n_u8` to right shift 4 bits and get the upper half
                // (3) use `vreinterpretq_s8_u8` to interpret the  vector as int8
                // lowbit mask
                const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
                const uint8x16_t w_low = vandq_u8(w0, mask_low4bit);
                const uint8x16_t w_high = vandq_u8(vshrq_n_u8(w0, 4), mask_low4bit);
                const int8x16_t w_low_s8 = vreinterpretq_s8_u8(w_low);
                const int8x16_t w_high_s8 = vreinterpretq_s8_u8(w_high);

                // TODO: apply zero_point to weights and convert the range from (0, 15) to (-8, 7)
                // Hint: using `vsubq_s8` to the lower-half and upper-half vectors of weights
                const int8x16_t offsets = vdupq_n_s8(8);
                const int8x16_t w_low_final = vsubq_s8(w_low_s8, offsets);
                const int8x16_t w_high_final = vsubq_s8(w_high_s8, offsets);

                // load 32 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                a_start += 32;

                // TODO: perform dot product and store the result into the intermediate sum, int_sum0
                // Hint: use `vdotq_s32` to compute sumv0 = a0 * lower-half weights + a1 * upper-half weights
                // int32x4 vector to store intermediate sum
                int32x4_t int_sum0 = vdupq_n_s32(0);
                int_sum0 = vdotq_s32(int_sum0, a0, w_low_final);
                int_sum0 = vdotq_s32(int_sum0, a1, w_high_final);

                float s_0 = *s_a++ * *s_w++;
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
            }
            C->data_ptr[row * n + col] = vaddvq_f32(sumv0);
#endif
#ifdef QM_x86
            // order of weights with QM_x86:
            // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
            // QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
            //               |--|
            //               4 bits
            //               |------|
            //               8 bits (byte)
            //            low|----------------------------------------------------------|high
            //               0                         256 bit
            __m256 acc0 = _mm256_setzero_ps();
            // pointer of the int4 weights
            const __m256i *w_start = (__m256i *)&B->int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const __m256i *a_start = (__m256i *)&A->int8_data_ptr[row * k];
            // scale of weight
            float *s_ptr = &params->scales[col * k / 32];
            // scale of activation
            float *sa_ptr = &params->A_scales[row * k / 32];

            const int num_block = k / block_size;
            // Compute two blocks in each iteration
            for (int q = 0; q < num_block; q += 2) {
                // lowbit mask
                const __m256i lowMask = _mm256_set1_epi8(0xF);

                /*
                   We will accelerate the program using x86 Intrinsics. You can check the documentation of operations
                   at: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avxnewtechs=AVX2
                */
                // TODO: Unpack 64 4-bit (one __mm256i) weights into 64 8-bit (two __mm256i)
                // (1) load 256 bit from w_strat with _mm256_loadu_si256
                // (2) use `_mm256_and_si256` and lowMask to extract the lower half of wegihts
                // (3) use `_mm256_srli_epi16` and `_mm256_and_si256` with lowMask to extract the upper half of weights
                __m256i raw_w = _mm256_loadu_si256(w_start);

                // TODO: apply zero_point to weights and convert the range from (0, 15) to (-8, 7)
                // Hint: using `_mm256_sub_epi8` to the lower-half and upper-half vectors of weights
                // Note: Store the lower half and upper half of weights into `w_0` and `w_128`, respectively
                const __m256i zero_point = _mm256_set1_epi8(8);
                __m256i w_0, w_128;

                // Perform int8 dot product with _mm256_maddubs_epi16
                /* Syntax of _mm256_maddubs_epi16:
                   __m256i _mm256_maddubs_epi16(__m256i s1, __m256i s2): Multiplies vertically each unsigned byte of
                   source vector s1 with the corresponding signed byte of source vector s2, producing intermediate,
                   signed 16-bit integers. Each adjacent pair of signed words is added, and the saturated result is
                   packed to the destination vector.
                */
                // To utilize _mm256_maddubs_epi16 which only takes unsigned s1, we need to:
                // (1) Get the absolute values of weights (for both lower and upper halves)
                // (2) Change the sign of activation (a0-a31 and a32-a63) depending on the sign of corresponding weights
                // (stored as another variable) (3) Perform dot product with _mm256_maddubs_epi16 and store the lower
                // and upper halves sum in `dot` and `dot2`
                __m256i dot, dot2;
                // Get absolute values of x vectors
                const __m256i ax = _mm256_sign_epi8(w_0, w_0);
                const __m256i ax2 = _mm256_sign_epi8(w_128, w_128);
                // Load activation
                __m256i activation = a_start[0];
                __m256i activation2 = a_start[1];
                // Sign the values of the y vectors
                const __m256i sy = _mm256_sign_epi8(activation, w_0);
                const __m256i sy2 = _mm256_sign_epi8(activation2, w_128);

                // TODO: Perform int8 dot product with `_mm256_maddubs_epi16`
                // Hint: use `_mm256_maddubs_epi16` to complete the following computation
                // dot = ax * sy
                // dot2 = ax2 * sy2

                // Convert int32 vectors to floating point vectors
                const __m256i ones = _mm256_set1_epi16(1);
                const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);
                const __m256i summed_pairs2 = _mm256_madd_epi16(ones, dot2);
                __m256 intermediate = _mm256_cvtepi32_ps(summed_pairs);
                __m256 intermediate2 = _mm256_cvtepi32_ps(summed_pairs2);

                // Create vectors for scales and apply them to intermediate results
                __m256 v_s = _mm256_set1_ps(s_ptr[0] * sa_ptr[0]);
                __m256 v_s2 = _mm256_set1_ps(s_ptr[1] * sa_ptr[1]);
                acc0 = _mm256_fmadd_ps(intermediate, v_s, acc0);
                acc0 = _mm256_fmadd_ps(intermediate2, v_s2, acc0);
                s_ptr += 2;
                sa_ptr += 2;
                w_start += 1;
                a_start += 2;
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
#endif
        }
    }
};
}  // namespace matmul
