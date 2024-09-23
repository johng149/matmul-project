#include <stdio.h>
#include <stdlib.h>

// A's `i`th row and `j`th column
#define A(i, j) a[i * M + j]
// A transpose's `i`th row and `j`th column
#define AT(i, j) a[j * M + i]
// B's `i`th row and `j`th column
#define B(i, j) b[i * M + j]
// B transpose's `i`th row and `j`th column
#define BT(i, j) b[j * M + i]
// C's `i`th row and `j`th column
#define C(i, j) c[i * M + j]

/*
    kernel_4x4_4x4 calculates the 4x4 block of C that is the result of multiplying the 4x4 block of A with the 4x4 block of B.

    M: size of original matrix
    A: pointer already moved to first element of the 4x4 block of A
    B: pointer already moved to first element of the 4x4 block of B
    C: pointer already moved to first element of the 4x4 block of C
*/
void kernel_4x4_4x4(const int M, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2), c13 = C(1, 3);
    double c20 = C(2, 0), c21 = C(2, 1), c22 = C(2, 2), c23 = C(2, 3);
    double c30 = C(3, 0), c31 = C(3, 1), c32 = C(3, 2), c33 = C(3, 3);

    for (int k = 0; k < 4; ++k)
    {
        c00 += A(0, k) * B(k, 0);
        c01 += A(0, k) * B(k, 1);
        c02 += A(0, k) * B(k, 2);
        c03 += A(0, k) * B(k, 3);

        c10 += A(1, k) * B(k, 0);
        c11 += A(1, k) * B(k, 1);
        c12 += A(1, k) * B(k, 2);
        c13 += A(1, k) * B(k, 3);

        c20 += A(2, k) * B(k, 0);
        c21 += A(2, k) * B(k, 1);
        c22 += A(2, k) * B(k, 2);
        c23 += A(2, k) * B(k, 3);

        c30 += A(3, k) * B(k, 0);
        c31 += A(3, k) * B(k, 1);
        c32 += A(3, k) * B(k, 2);
        c33 += A(3, k) * B(k, 3);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;
    C(0, 3) = c03;

    C(1, 0) = c10;
    C(1, 1) = c11;
    C(1, 2) = c12;
    C(1, 3) = c13;

    C(2, 0) = c20;
    C(2, 1) = c21;
    C(2, 2) = c22;
    C(2, 3) = c23;

    C(3, 0) = c30;
    C(3, 1) = c31;
    C(3, 2) = c32;
    C(3, 3) = c33;
}

void kernel_4x4_4x1(const int M, double *restrict a, double *restrict b, double *c)
{
    double c0 = C(0, 0), c1 = C(1, 0), c2 = C(2, 0), c3 = C(3, 0);

    for (int k = 0; k < 4; ++k)
    {
        c0 += A(0, k) * B(k, 0);
        c1 += A(1, k) * B(k, 0);
        c2 += A(2, k) * B(k, 0);
        c3 += A(3, k) * B(k, 0);
    }

    C(0, 0) = c0;
    C(1, 0) = c1;
    C(2, 0) = c2;
    C(3, 0) = c3;
}

void kernel_4x4_4x2(const int M, double *restrict a, double *restrict b, double *c)
{
    double c0 = C(0, 0), c1 = C(1, 0), c2 = C(2, 0), c3 = C(3, 0);
    double c4 = C(0, 1), c5 = C(1, 1), c6 = C(2, 1), c7 = C(3, 1);

    for (int k = 0; k < 4; ++k)
    {
        c0 += A(0, k) * B(k, 0);
        c1 += A(1, k) * B(k, 0);
        c2 += A(2, k) * B(k, 0);
        c3 += A(3, k) * B(k, 0);

        c4 += A(0, k) * B(k, 1);
        c5 += A(1, k) * B(k, 1);
        c6 += A(2, k) * B(k, 1);
        c7 += A(3, k) * B(k, 1);
    }

    C(0, 0) = c0;
    C(1, 0) = c1;
    C(2, 0) = c2;
    C(3, 0) = c3;

    C(0, 1) = c4;
    C(1, 1) = c5;
    C(2, 1) = c6;
    C(3, 1) = c7;
}

void kernel_4x4_4x3(const int M, double *restrict a, double *restrict b, double *c)
{
    double c0 = C(0, 0), c1 = C(1, 0), c2 = C(2, 0), c3 = C(3, 0);
    double c4 = C(0, 1), c5 = C(1, 1), c6 = C(2, 1), c7 = C(3, 1);
    double c8 = C(0, 2), c9 = C(1, 2), c10 = C(2, 2), c11 = C(3, 2);

    for (int k = 0; k < 4; ++k)
    {
        c0 += A(0, k) * B(k, 0);
        c1 += A(1, k) * B(k, 0);
        c2 += A(2, k) * B(k, 0);
        c3 += A(3, k) * B(k, 0);

        c4 += A(0, k) * B(k, 1);
        c5 += A(1, k) * B(k, 1);
        c6 += A(2, k) * B(k, 1);
        c7 += A(3, k) * B(k, 1);

        c8 += A(0, k) * B(k, 2);
        c9 += A(1, k) * B(k, 2);
        c10 += A(2, k) * B(k, 2);
        c11 += A(3, k) * B(k, 2);
    }

    C(0, 0) = c0;
    C(1, 0) = c1;
    C(2, 0) = c2;
    C(3, 0) = c3;

    C(0, 1) = c4;
    C(1, 1) = c5;
    C(2, 1) = c6;
    C(3, 1) = c7;

    C(0, 2) = c8;
    C(1, 2) = c9;
    C(2, 2) = c10;
    C(3, 2) = c11;
}

void kernel_4x1_1x4(const int M, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2), c13 = C(1, 3);
    double c20 = C(2, 0), c21 = C(2, 1), c22 = C(2, 2), c23 = C(2, 3);
    double c30 = C(3, 0), c31 = C(3, 1), c32 = C(3, 2), c33 = C(3, 3);

    c00 += A(0, 0) * B(0, 0);
    c01 += A(0, 0) * B(0, 1);
    c02 += A(0, 0) * B(0, 2);
    c03 += A(0, 0) * B(0, 3);

    c10 += A(1, 0) * B(0, 0);
    c11 += A(1, 0) * B(0, 1);
    c12 += A(1, 0) * B(0, 2);
    c13 += A(1, 0) * B(0, 3);

    c20 += A(2, 0) * B(0, 0);
    c21 += A(2, 0) * B(0, 1);
    c22 += A(2, 0) * B(0, 2);
    c23 += A(2, 0) * B(0, 3);

    c30 += A(3, 0) * B(0, 0);
    c31 += A(3, 0) * B(0, 1);
    c32 += A(3, 0) * B(0, 2);
    c33 += A(3, 0) * B(0, 3);

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;
    C(0, 3) = c03;

    C(1, 0) = c10;
    C(1, 1) = c11;
    C(1, 2) = c12;
    C(1, 3) = c13;

    C(2, 0) = c20;
    C(2, 1) = c21;
    C(2, 2) = c22;
    C(2, 3) = c23;

    C(3, 0) = c30;
    C(3, 1) = c31;
    C(3, 2) = c32;
    C(3, 3) = c33;
}

void kernel_4x2_2x4(const int M, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2), c13 = C(1, 3);
    double c20 = C(2, 0), c21 = C(2, 1), c22 = C(2, 2), c23 = C(2, 3);
    double c30 = C(3, 0), c31 = C(3, 1), c32 = C(3, 2), c33 = C(3, 3);

    for (int i = 0; i < 2; ++i)
    {
        c00 += A(0, i) * B(i, 0);
        c01 += A(0, i) * B(i, 1);
        c02 += A(0, i) * B(i, 2);
        c03 += A(0, i) * B(i, 3);
    }

    for (int i = 0; i < 2; ++i)
    {
        c10 += A(1, i) * B(i, 0);
        c11 += A(1, i) * B(i, 1);
        c12 += A(1, i) * B(i, 2);
        c13 += A(1, i) * B(i, 3);
    }

    for (int i = 0; i < 2; ++i)
    {
        c20 += A(2, i) * B(i, 0);
        c21 += A(2, i) * B(i, 1);
        c22 += A(2, i) * B(i, 2);
        c23 += A(2, i) * B(i, 3);
    }

    for (int i = 0; i < 2; ++i)
    {
        c30 += A(3, i) * B(i, 0);
        c31 += A(3, i) * B(i, 1);
        c32 += A(3, i) * B(i, 2);
        c33 += A(3, i) * B(i, 3);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;
    C(0, 3) = c03;

    C(1, 0) = c10;
    C(1, 1) = c11;
    C(1, 2) = c12;
    C(1, 3) = c13;

    C(2, 0) = c20;
    C(2, 1) = c21;
    C(2, 2) = c22;
    C(2, 3) = c23;

    C(3, 0) = c30;
    C(3, 1) = c31;
    C(3, 2) = c32;
    C(3, 3) = c33;
}

void kernel_4x3_3x4(const int M, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2), c13 = C(1, 3);
    double c20 = C(2, 0), c21 = C(2, 1), c22 = C(2, 2), c23 = C(2, 3);
    double c30 = C(3, 0), c31 = C(3, 1), c32 = C(3, 2), c33 = C(3, 3);

    for (int i = 0; i < 3; ++i)
    {
        c00 += A(0, i) * B(i, 0);
        c01 += A(0, i) * B(i, 1);
        c02 += A(0, i) * B(i, 2);
        c03 += A(0, i) * B(i, 3);
    }

    for (int i = 0; i < 3; ++i)
    {
        c10 += A(1, i) * B(i, 0);
        c11 += A(1, i) * B(i, 1);
        c12 += A(1, i) * B(i, 2);
        c13 += A(1, i) * B(i, 3);
    }

    for (int i = 0; i < 3; ++i)
    {
        c20 += A(2, i) * B(i, 0);
        c21 += A(2, i) * B(i, 1);
        c22 += A(2, i) * B(i, 2);
        c23 += A(2, i) * B(i, 3);
    }

    for (int i = 0; i < 3; ++i)
    {
        c30 += A(3, i) * B(i, 0);
        c31 += A(3, i) * B(i, 1);
        c32 += A(3, i) * B(i, 2);
        c33 += A(3, i) * B(i, 3);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;
    C(0, 3) = c03;

    C(1, 0) = c10;
    C(1, 1) = c11;
    C(1, 2) = c12;
    C(1, 3) = c13;

    C(2, 0) = c20;
    C(2, 1) = c21;
    C(2, 2) = c22;
    C(2, 3) = c23;

    C(3, 0) = c30;
    C(3, 1) = c31;
    C(3, 2) = c32;
    C(3, 3) = c33;
}

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

void square_dgemm(const int M, double *a, double *b, double *c)
{
    kernel_4x4_4x4(M, a, b, c);
}

void main()
{
    const int M = 5;
    double *a = (double *)malloc(M * M * sizeof(double));
    double *b = (double *)malloc(M * M * sizeof(double));
    double *c = (double *)malloc(M * M * sizeof(double));
    for (int i = 0; i < M * M; ++i)
    {
        a[i] = i;
        b[i] = i * -1;
    }

    const int start_row_a = 1;
    const int start_col_a = 1;
    const int start_row_b = 1;
    const int start_col_b = 1;
    const int write_row = 1;
    const int write_col = 1;

    double *a_4x3 = &A(start_row_a, start_col_a);
    double *b_3x4 = &B(start_row_b, start_col_b);
    double *c_4x4 = &C(write_row, write_col);

    kernel_4x3_3x4(M, a_4x3, b_3x4, c_4x4);

    // print result
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            printf("%f ", C(i, j));
        }
        printf("\n");
    }
}