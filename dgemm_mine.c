#include <stdlib.h>

const char *dgemm_desc = "My dgemm.";

// A's `i`th row and `j`th column
#define A(i, j) A[i * M + j]
// A transpose's `i`th row and `j`th column
#define AT(i, j) A[j * M + i]
// B's `i`th row and `j`th column
#define B(i, j) B[i * M + j]
// B transpose's `i`th row and `j`th column
#define BT(i, j) B[j * M + i]
// C's `i`th row and `j`th column
#define C(i, j) C[i * M + j]

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

void square_dgemm_helper(const int M, double *A, double *B, double *C)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            C(i, j) = 0;
            for (int k = 0; k < M; ++k)
            {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

void square_dgemm(const int M, double *A, double *B, double *C)
{
    square_dgemm_helper(M, B, A, C);
}