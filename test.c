#include <immintrin.h>
#include <stdio.h>

void vector_outer_product(__m256d a0, __m256d b0, __m256d *result)
{
    // Broadcast each element of a0 into a separate vector
    __m256d a0_broadcasted0 = _mm256_set1_pd(((double *)&a0)[0]);
    __m256d a0_broadcasted1 = _mm256_set1_pd(((double *)&a0)[1]);
    __m256d a0_broadcasted2 = _mm256_set1_pd(((double *)&a0)[2]);
    __m256d a0_broadcasted3 = _mm256_set1_pd(((double *)&a0)[3]);

    // Multiply the broadcasted vectors with b0
    result[0] = _mm256_mul_pd(a0_broadcasted0, b0);
    result[1] = _mm256_mul_pd(a0_broadcasted1, b0);
    result[2] = _mm256_mul_pd(a0_broadcasted2, b0);
    result[3] = _mm256_mul_pd(a0_broadcasted3, b0);
}

int main()
{
    __m256d a0 = {1, 2, 3, 4};
    __m256d b0 = {-3, -2, -4, -1};
    __m256d result[4];

    vector_outer_product(a0, b0, result);

    // Print the result (adjust formatting as needed)
    for (int i = 0; i < 4; ++i)
    {
        printf("Row %d: %.2f %.2f %.2f %.2f\n", i,
               ((double *)&result[i])[0], ((double *)&result[i])[1],
               ((double *)&result[i])[2], ((double *)&result[i])[3]);
    }

    int row = 4;
    int col = 4;
    double *C = (double *)calloc(row * col, sizeof(double));

    int write_row = 0;
    int write_col = 0;
    int out_rows = 3;
    int out_cols = 3;
    for (int i = 0; i < out_rows; ++i)
    {
        for (int j = 0; j < out_cols; ++j)
        {
            int out_flat = (write_row + i) * row + j + write_col;
            C[out_flat] += ((double *)&result[i])[j];
        }
    }

    for (int i = 0; i < row * col; ++i)
    {
        printf("%.2f ", C[i]);
    }

    return 0;
}