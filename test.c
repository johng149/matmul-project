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

void main()
{
    const int M = 4;
    double *A = (double *)malloc(M * M * sizeof(double));
    double *B = (double *)malloc(M * M * sizeof(double));
    double *C = (double *)malloc(M * M * sizeof(double));
    for (int i = 0; i < M * M; ++i)
    {
        A[i] = i;
        B[i] = i * -1;
    }
    double *BT = transpose(M, B);
}