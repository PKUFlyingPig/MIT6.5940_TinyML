#include <cassert>

#ifdef QM_ARM
#include <arm_neon.h>
void quantize_fp32_to_int8(float* A, int8_t* qA, float* sA, int size, int block_size) {
    assert(size % block_size == 0);
    assert(block_size == 32);
    int num_block = size / 32;

    for (int i = 0; i < num_block; i++) {
        float32x4_t srcv[8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        int8_t* start_qA = &qA[i * 32];

        for (int l = 0; l < 8; l++) srcv[l] = vld1q_f32(A + i * 32 + 4 * l);
        for (int l = 0; l < 8; l++) asrcv[l] = vabsq_f32(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2 * l] = vmaxq_f32(asrcv[2 * l], asrcv[2 * l + 1]);
        for (int l = 0; l < 2; l++) amaxv[4 * l] = vmaxq_f32(amaxv[4 * l], amaxv[4 * l + 2]);
        for (int l = 0; l < 1; l++) amaxv[8 * l] = vmaxq_f32(amaxv[8 * l], amaxv[8 * l + 4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        sA[i] = d;

        // low half
        for (int l = 0; l < 4; l++) {
            const float32x4_t v = vmulq_n_f32(srcv[l], id);
            const int32x4_t vi = vcvtnq_s32_f32(v);

            start_qA[4 * l + 0] = vgetq_lane_s32(vi, 0);
            start_qA[4 * l + 1] = vgetq_lane_s32(vi, 1);
            start_qA[4 * l + 2] = vgetq_lane_s32(vi, 2);
            start_qA[4 * l + 3] = vgetq_lane_s32(vi, 3);
        }

        // high half
        for (int l = 4; l < 8; l++) {
            const float32x4_t v = vmulq_n_f32(srcv[l], id);
            const int32x4_t vi = vcvtnq_s32_f32(v);

            start_qA[4 * l + 0] = vgetq_lane_s32(vi, 0);
            start_qA[4 * l + 1] = vgetq_lane_s32(vi, 1);
            start_qA[4 * l + 2] = vgetq_lane_s32(vi, 2);
            start_qA[4 * l + 3] = vgetq_lane_s32(vi, 3);
        }
    }
}
#endif
#ifdef QM_x86
#include <immintrin.h>
void quantize_fp32_to_int8(float* A, int8_t* qA, float* sA, int size, int block_size) {
    int nb = size / 32;
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(A);
        __m256 v1 = _mm256_loadu_ps(A + 8);
        __m256 v2 = _mm256_loadu_ps(A + 16);
        __m256 v3 = _mm256_loadu_ps(A + 24);
        A += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        // Quantize these floats
        const float d = maxScalar / 127.f;
        *sA++ = d;
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);

        // Apply the multiplier
        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        // Round to nearest integer
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        // Convert int32 to int16
        i0 = _mm256_packs_epi32(i0, i1);  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32(i2, i3);  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                          // Convert int16 to int8
        i0 = _mm256_packs_epi16(i0, i2);  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7,
                                          // 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);

        _mm256_storeu_si256((__m256i*)qA, i0);
        qA += 32;
    }
}
#endif
