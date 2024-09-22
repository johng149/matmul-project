#include <math.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>

// since our smallest kernel is a 4 by 4 we define the constants accordingly
#define SUBPARTITION_SIZE 4 // rows
#define PANEL_BLOCK_SIZE 4  // cols

const char *dgemm_desc = "My awesome dgemm.";

void vector_outer_product(__m256d a0, __m256d b0, __m256d *result)
{

    // Broadcast each element of a0 into a separate vector
    __m256d a0_broadcasted0 = _mm256_set1_pd(((double *)&a0)[0]);
    __m256d a0_broadcasted1 = _mm256_set1_pd(((double *)&a0)[1]);
    __m256d a0_broadcasted2 = _mm256_set1_pd(((double *)&a0)[2]);
    __m256d a0_broadcasted3 = _mm256_set1_pd(((double *)&a0)[3]);

    // Multiply the broadcasted vectors with b0
    result[0] = _mm256_add_pd(result[0], _mm256_mul_pd(a0_broadcasted0, b0));
    result[1] = _mm256_add_pd(result[1], _mm256_mul_pd(a0_broadcasted1, b0));
    result[2] = _mm256_add_pd(result[2], _mm256_mul_pd(a0_broadcasted2, b0));
    result[3] = _mm256_add_pd(result[3], _mm256_mul_pd(a0_broadcasted3, b0));
}

/*
    A_packed has 16 elements as does B_packed, they are aligned in memory and already
    have padding zeros as needed
*/
void vector_outer_product_4x4(const double *restrict A_packed, const double *restrict B_packed, __m256d *result)
{
    __m256d a0 = _mm256_load_pd(A_packed);
    __m256d a1 = _mm256_load_pd(A_packed + 4);
    __m256d a2 = _mm256_load_pd(A_packed + 8);
    __m256d a3 = _mm256_load_pd(A_packed + 12);

    __m256d b0 = _mm256_load_pd(B_packed);
    __m256d b1 = _mm256_load_pd(B_packed + 4);
    __m256d b2 = _mm256_load_pd(B_packed + 8);
    __m256d b3 = _mm256_load_pd(B_packed + 12);

    vector_outer_product(a0, b0, result);
    vector_outer_product(a1, b1, result);
    vector_outer_product(a2, b2, result);
    vector_outer_product(a3, b3, result);
}

/*
    Takes in a transposed matrix, along with where to start packing, and creates an aligned packed matrix.
    Will pad with zeros as needed

    The input matrix has `r` rows and `c` columns
*/
void pack_matrix_t(
    const double *restrict AT,
    const int r,
    const int c,
    const int start_row,
    const int start_col,
    const int rows_to_pack,
    const int cols_to_pack,
    const int out_rows,
    const int out_cols,
    double *packed_matrix // given should have out_rows * out_cols elements
)
{
    // double *packed_matrix = (double *)aligned_alloc(32, out_rows * out_cols * sizeof(double));
    // memset(packed_matrix, 0, out_rows * out_cols * sizeof(double));

    for (int i = 0; i < out_cols; ++i)
    {
        for (int j = 0; j < out_rows; ++j)
        {
            const int at_idx = (start_col + i) * r + start_row + j;
            const int packed_idx = i * out_rows + j;
            if (j >= rows_to_pack || i >= cols_to_pack)
            {
                packed_matrix[packed_idx] = 0;
            }
            else
            {
                packed_matrix[packed_idx] = AT[at_idx];
            }
        }
    }

    // return packed_matrix;
}

void pack_matrix(
    const double *restrict A,
    const int r,
    const int c,
    const int start_row,
    const int start_col,
    const int rows_to_pack,
    const int cols_to_pack,
    const int out_rows,
    const int out_cols,
    double *packed_matrix // given should have out_rows * out_cols elements
)
{
    // double *packed_matrix = (double *)aligned_alloc(32, out_rows * out_cols * sizeof(double));
    // memset(packed_matrix, 0, out_rows * out_cols * sizeof(double));

    for (int i = 0; i < out_rows; ++i)
    {
        for (int j = 0; j < out_cols; ++j)
        {
            const int a_idx = (start_row + i) * c + start_col + j;
            const int packed_idx = i * out_cols + j;
            if (j >= cols_to_pack || i >= rows_to_pack)
            {
                packed_matrix[packed_idx] = 0;
            }
            else
            {
                packed_matrix[packed_idx] = A[a_idx];
            }
        }
    }

    // return packed_matrix;
}

void write_out(
    __m256d *result,
    double *C,
    const int r,
    const int c,
    const int write_row,
    const int write_col,
    const int out_rows,
    const int out_cols)
{
    for (int i = 0; i < out_rows; ++i)
    {
        for (int j = 0; j < out_cols; ++j)
        {
            const int c_idx = (write_row + i) * c + write_col + j;
            C[c_idx] = ((double *)&result[i])[j] + C[c_idx];
        }
    }
}

// https://www.geeksforgeeks.org/compute-the-minimum-or-maximum-max-of-two-integers-without-branching/
int min(int x, int y)
{
    return y ^ ((x ^ y) & -(x < y));
}

/*
    X is N by N matrix stored in row-major order in 1D

    Return the transpose of X
*/
double *transpose(const int N, const double *X)
{
    double *X_T = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int original = i * N + j;
            int transposed = j * N + i;
            X_T[transposed] = X[original];
        }
    }
    return X_T;
}

/*
    A is M by M matrix stored in column-major order in 1D
    B is M by M matrix stored in row-major order in 1D
    C is M by M matrix stored in column-major order in 1D
*/
void panel_panel_dgemm_a_t(
    const int M,
    const int panel_a_rows,
    const int panel_a_cols,
    const int panel_b_rows,
    const int panel_b_cols,
    const int start_row_a,
    const int start_col_a,
    const int start_row_b,
    const int start_col_b,
    const int write_row,
    const int write_col,
    const double *restrict AT,
    const double *restrict B,
    double *C,
    const double *restrict A_packed,
    const double *restrict B_packed)
{
    __m256d result[4] = {
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd()};

    pack_matrix_t(AT, M, M, start_row_a, start_col_a, panel_a_rows, panel_a_cols, PANEL_BLOCK_SIZE, SUBPARTITION_SIZE, A_packed);
    pack_matrix(B, M, M, start_row_b, start_col_b, panel_b_rows, panel_b_cols, PANEL_BLOCK_SIZE, SUBPARTITION_SIZE, B_packed);

    vector_outer_product_4x4(A_packed, B_packed, result);

    const int out_rows = panel_a_rows;
    const int out_cols = panel_b_cols;

    write_out(result, C, M, M, write_row, write_col, out_rows, out_cols);
}

/*
    A is M by M matrix stored in column-major order in 1D
    B is M by M matrix stored in row-major order in 1D
    C is M by M matrix stored in column-major order in 1D
*/
void panel_panel_dgemm_recurse_a_t(
    const int M,
    const int panel_a_rows,
    const int panel_a_cols,
    const int panel_b_rows,
    const int panel_b_cols,
    const int start_row_a,
    const int start_col_a,
    const int start_row_b,
    const int start_col_b,
    const int write_row,
    const int write_col,
    const double *restrict AT,
    const double *restrict B,
    double *C,
    int block_size,
    const double *restrict a_packed,
    const double *restrict b_packed)
{
    const int loop_cap = (block_size < M) * ((int)ceil((double)M / block_size) - 1);
    for (int i = 0; i <= loop_cap; ++i)
    {
        int start_col_a_i = start_col_a + i * block_size;
        int start_row_b_i = start_col_a_i;
        int panel_a_cols_i = min(block_size, M - start_col_a_i);
        int panel_b_rows_i = panel_a_cols_i;
        panel_panel_dgemm_a_t(
            M,
            panel_a_rows,
            panel_a_cols_i,
            panel_b_rows_i,
            panel_b_cols,
            start_row_a,
            start_col_a_i,
            start_row_b_i,
            start_col_b,
            write_row,
            write_col,
            AT,
            B,
            C, a_packed, b_packed);
    }
}

void square_dgemm_helper(
    const int M,
    const double *restrict A,
    const double *restrict B,
    double *C,
    const int panel_block_size,
    const int subpartition_size)
{
    const int start_col_a = 0;
    const int start_row_b = 0;
    const int panel_a_cols = M;
    const int panel_b_rows = M;
    const double *AT = transpose(M, A);
    const int loop_cap = (panel_block_size < M) * ((int)ceil((double)M / panel_block_size) - 1);
    double *b_packed = (double *)aligned_alloc(32, PANEL_BLOCK_SIZE * SUBPARTITION_SIZE * sizeof(double));
    double *a_packed = (double *)aligned_alloc(32, PANEL_BLOCK_SIZE * SUBPARTITION_SIZE * sizeof(double));
    for (int i = 0; i <= loop_cap; ++i)
    {
        int start_row_a = i * panel_block_size;
        int panel_a_rows = min(panel_block_size, M - start_row_a);
        for (int j = 0; j <= loop_cap; ++j)
        {
            int start_col_b = j * panel_block_size;
            int panel_b_cols = min(panel_block_size, M - start_col_b);
            int write_row = start_row_a;
            int write_col = start_col_b;
            panel_panel_dgemm_recurse_a_t(
                M,
                panel_a_rows,
                panel_a_cols,
                panel_b_rows,
                panel_b_cols,
                start_row_a,
                start_col_a,
                start_row_b,
                start_col_b,
                write_row,
                write_col,
                AT,
                B,
                C,
                subpartition_size, a_packed, b_packed);
        }
    }

    free((void *)a_packed);
    free((void *)b_packed);
    free(AT);
}

void square_dgemm(const int M, const double *restrict A, const double *restrict B, double *C)
{
    square_dgemm_helper(M, B, A, C, PANEL_BLOCK_SIZE, SUBPARTITION_SIZE);
}