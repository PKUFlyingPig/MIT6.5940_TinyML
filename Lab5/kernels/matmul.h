#include <stdint.h>
#include <sys/time.h>

void quantize_fp32_to_int8(float *A, int8_t *qA, float *sA, int size, int block_size);

// Data structures
struct quantization_params {
    float scale;
    bool per_channel = false;
    int32_t zero_point;
    int8_t q_min = -128, q_max = 127;
};

struct matrix {
    int row;
    int column;
    float *data_ptr;
    int32_t *int32_data_ptr;
    int8_t *int8_data_ptr;
    uint8_t *uint8_data_ptr;
    uint8_t *int4_data_ptr;
    struct quantization_params qparams;
    int length() { return row * column; }
};

struct optimization_params {
    int blk_size;
    int num_thread = 8;
};

struct matmul_params {
    struct matrix A, B, C, bias;
    struct optimization_params opt_params;
    float alpha, beta;
    // for int4
    float *scales, *offset, *zero_point;
    int block_size;
    // for int8 activation
    float *A_scales;
    int8_t A_zero_point;
};

struct thread_args {
    const struct matrix *A;
    const struct matrix *B;
    const struct matrix *C;
    const struct matmul_params *params;
    int start_i, end_i, blk_size;
};

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))
namespace matmul {
class MatmulOperator {
   public:
    void mat_mul_transposed(const struct matmul_params *params);
    void mat_mul_accelerator_transposed_fastover_column(const struct matmul_params *params);
    // int8
    void naive_mat_mul_int8(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(const struct matmul_params *params);
    // int4
    void mat_mul_accelerator_int4_fast(const struct matmul_params *params);
    void mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params *params);
    void mat_mul_accelerator_int8_int4_fast_no_offset(struct matmul_params *params);
    void naive_mat_mul_int4(const struct matmul_params *params);
    void naive_mat_mul_int4_with_offset(const struct matmul_params *params);
    // w8a4 code template functions
    void mat_mul_reference(struct matmul_params *params);
    void mat_mul_loop_unrolling(struct matmul_params *params);
    void mat_mul_multithreading(struct matmul_params *params);
    void mat_mul_multithreading_loop_unrolling(struct matmul_params *params);
    void mat_mul_simd_programming(struct matmul_params *params);
    void mat_mul_all_techniques(struct matmul_params *params);
    // cuda
    void mat_mul_cuda(const struct matmul_params *params);

   private:
    float interval_to_us(struct timeval *start, struct timeval *end);
    void CHECK_MATRICES(const struct matrix *A, const struct matrix *B, const struct matrix *C);
    void CHECK_MATRICES_int4weight(const struct matrix *A, const struct matrix *B, const struct matrix *C);
};
}  // namespace matmul
