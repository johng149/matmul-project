const char *dgemm_desc = "My awesome dgemm.";

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

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k;
    double *B_T = transpose(M, B);
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < M; ++j)
        {
            double cij = C[j * M + i];
            for (k = 0; k < M; ++k)
                cij += A[k * M + i] * B[j * M + k];
            C[j * M + i] = cij;
        }
    }
}