#include <immintrin.h>
#include <stdio.h>
#include <string.h>

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
double *pack_matrix_t(
    const double *restrict AT,
    const int r,
    const int c,
    const int start_row,
    const int start_col,
    const int rows_to_pack,
    const int cols_to_pack,
    const int out_rows,
    const int out_cols)
{
    double *packed_matrix = (double *)aligned_alloc(32, out_rows * out_cols * sizeof(double));
    memset(packed_matrix, 0, out_rows * out_cols * sizeof(double));

    for (int i = 0; i < cols_to_pack; ++i)
    {
        for (int j = 0; j < rows_to_pack; ++j)
        {
            const int at_idx = (start_col + i) * r + start_row + j;
            const int packed_idx = i * out_rows + j;
            packed_matrix[packed_idx] = AT[at_idx];
        }
    }

    return packed_matrix;
}

double *pack_matrix(
    const double *restrict A,
    const int r,
    const int c,
    const int start_row,
    const int start_col,
    const int rows_to_pack,
    const int cols_to_pack,
    const int out_rows,
    const int out_cols)
{
    double *packed_matrix = (double *)aligned_alloc(32, out_rows * out_cols * sizeof(double));
    memset(packed_matrix, 0, out_rows * out_cols * sizeof(double));

    for (int i = 0; i < rows_to_pack; ++i)
    {
        for (int j = 0; j < cols_to_pack; ++j)
        {
            const int a_idx = (start_row + i) * c + start_col + j;
            const int packed_idx = i * out_cols + j;
            packed_matrix[packed_idx] = A[a_idx];
        }
    }

    return packed_matrix;
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
            C[c_idx] = ((double *)&result[i])[j];
        }
    }
}

int main()
{
    __m256d result[4] = {
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd()};

    const int r = 4;
    const int c = 4;

    double *A = (double *)calloc(r * c, sizeof(double));
    double *B = (double *)calloc(r * c, sizeof(double));
    double *C = (double *)calloc(r * c, sizeof(double));

    for (int i = 0; i < r * c; ++i)
    {
        A[i] = i;
        B[i] = i * -1;
        C[i] = -69;
    }

    double *AT = transpose(r, A);

    const int start_row_a = 0;
    const int start_col_a = 0;
    const int rows_to_pack_a = 3;
    const int cols_to_pack_a = 2;
    const int out_rows_a = 4;
    const int out_cols_a = 4;
    const int start_row_b = 0;
    const int start_col_b = 0;
    const int rows_to_pack_b = 2;
    const int cols_to_pack_b = 3;
    const int out_rows_b = 4;
    const int out_cols_b = 4;

    double *A_packed = pack_matrix_t(AT, r, c, start_row_a, start_col_a, rows_to_pack_a, cols_to_pack_a, out_rows_a, out_cols_a);
    double *B_packed = pack_matrix(B, r, c, start_row_b, start_col_b, rows_to_pack_b, cols_to_pack_b, out_rows_b, out_cols_b);

    // Print out what A_packed and B_packed look like
    printf("A_packed\n");
    for (int i = 0; i < 16; ++i)
    {
        printf("%.2f ", A_packed[i]);
    }

    printf("\nB_packed\n");

    for (int i = 0; i < 16; ++i)
    {
        printf("%.2f ", B_packed[i]);
    }

    printf("\nCalculating\n");

    vector_outer_product_4x4(A_packed, B_packed, result);

    // Print the result (adjust formatting as needed)
    for (int i = 0; i < 4; ++i)
    {
        printf("Row %d: %.2f %.2f %.2f %.2f\n", i,
               ((double *)&result[i])[0], ((double *)&result[i])[1],
               ((double *)&result[i])[2], ((double *)&result[i])[3]);
    }

    const write_row = 0;
    const write_col = 0;
    const out_rows = rows_to_pack_a;
    const out_cols = cols_to_pack_b;

    write_out(result, C, r, c, write_row, write_col, out_rows, out_cols);

    // print out C
    printf("\nC\n");
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            printf("%.2f ", C[i * c + j]);
        }
        printf("\n");
    }

    return 0;
}