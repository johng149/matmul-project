#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const char *dgemm_desc = "My awesome dgemm.";

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
    const double *AT,
    const double *B,
    double *C)
{
    for (int i = 0; i < panel_a_cols; ++i)
    {
        int a_c = start_col_a + i;
        int b_r = start_row_b + i;
        for (int k = 0; k < panel_a_rows; ++k)
        {
            int a_r = start_row_a + k;
            for (int j = 0; j < panel_b_cols; ++j)
            {
                int b_c = start_col_b + j;
                int a_flat = a_c * M + a_r;
                int b_flat = b_r * M + b_c;
                int out_flat = (write_row + k) * M + j + write_col;
                C[out_flat] += AT[a_flat] * B[b_flat];
            }
        }
    }
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
    const double *AT,
    const double *B,
    double *C,
    int block_size)
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
            C);
    }
}

void square_dgemm_helper(
    const int M,
    const double *A,
    const double *B,
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
                subpartition_size);
        }
    }
}

void square_dgemm(const int M, const double *restrict A, const double *restrict B, double *C)
{
    const int panel_block_size = 64;
    const int subpartition_size = 8;
    square_dgemm_helper(M, B, A, C, panel_block_size, subpartition_size);
}