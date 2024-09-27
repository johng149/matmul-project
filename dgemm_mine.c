#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

// A's `i`th row and `j`th column
#define A(i, j) a[i * Ma + j]
// A transpose's `i`th row and `j`th column
#define AT(i, j) a[j * Ma + i]
// B's `i`th row and `j`th column
#define B(i, j) b[i * Mb + j]
// B transpose's `i`th row and `j`th column
#define BT(i, j) b[j * Mb + i]
// C's `i`th row and `j`th column
#define C(i, j) c[i * Mc + j]

// start of kernels

/*
    kernel_4x4_4x4 calculates the 4x4 block of C that is the result of multiplying the 4x4 block of A with the 4x4 block of B.

    M: size of original matrix
    A: pointer already moved to first element of the 4x4 block of A
    B: pointer already moved to first element of the 4x4 block of B
    C: pointer already moved to first element of the 4x4 block of C
*/
void kernel_4x4_4x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    __m256d c0 = _mm256_loadu_pd(&C(0, 0));
    __m256d c1 = _mm256_loadu_pd(&C(1, 0));
    __m256d c2 = _mm256_loadu_pd(&C(2, 0));
    __m256d c3 = _mm256_loadu_pd(&C(3, 0));

    for (int i = 0; i < 4; ++i)
    {
        __m256d a0 = _mm256_set1_pd(A(0, i));
        __m256d a1 = _mm256_set1_pd(A(1, i));
        __m256d a2 = _mm256_set1_pd(A(2, i));
        __m256d a3 = _mm256_set1_pd(A(3, i));
        __m256d b0 = _mm256_loadu_pd(&B(i, 0));

        c0 = _mm256_fmadd_pd(a0, b0, c0);
        c1 = _mm256_fmadd_pd(a1, b0, c1);
        c2 = _mm256_fmadd_pd(a2, b0, c2);
        c3 = _mm256_fmadd_pd(a3, b0, c3);
    }
    _mm256_storeu_pd(&C(0, 0), c0);
    _mm256_storeu_pd(&C(1, 0), c1);
    _mm256_storeu_pd(&C(2, 0), c2);
    _mm256_storeu_pd(&C(3, 0), c3);
}

void kernel_8x8_8x8(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    __m512d c0 = _mm512_loadu_pd(&C(0, 0));
    __m512d c1 = _mm512_loadu_pd(&C(1, 0));
    __m512d c2 = _mm512_loadu_pd(&C(2, 0));
    __m512d c3 = _mm512_loadu_pd(&C(3, 0));
    __m512d c4 = _mm512_loadu_pd(&C(4, 0));
    __m512d c5 = _mm512_loadu_pd(&C(5, 0));
    __m512d c6 = _mm512_loadu_pd(&C(6, 0));
    __m512d c7 = _mm512_loadu_pd(&C(7, 0));

    for (int i = 0; i < 8; ++i)
    {
        __m512d a0 = _mm512_set1_pd(A(0, i));
        __m512d a1 = _mm512_set1_pd(A(1, i));
        __m512d a2 = _mm512_set1_pd(A(2, i));
        __m512d a3 = _mm512_set1_pd(A(3, i));
        __m512d a4 = _mm512_set1_pd(A(4, i));
        __m512d a5 = _mm512_set1_pd(A(5, i));
        __m512d a6 = _mm512_set1_pd(A(6, i));
        __m512d a7 = _mm512_set1_pd(A(7, i));
        __m512d b0 = _mm512_loadu_pd(&B(i, 0));

        c0 = _mm512_fmadd_pd(a0, b0, c0);
        c1 = _mm512_fmadd_pd(a1, b0, c1);
        c2 = _mm512_fmadd_pd(a2, b0, c2);
        c3 = _mm512_fmadd_pd(a3, b0, c3);
        c4 = _mm512_fmadd_pd(a4, b0, c4);
        c5 = _mm512_fmadd_pd(a5, b0, c5);
        c6 = _mm512_fmadd_pd(a6, b0, c6);
        c7 = _mm512_fmadd_pd(a7, b0, c7);
    }

    _mm512_storeu_pd(&C(0, 0), c0);
    _mm512_storeu_pd(&C(1, 0), c1);
    _mm512_storeu_pd(&C(2, 0), c2);
    _mm512_storeu_pd(&C(3, 0), c3);
    _mm512_storeu_pd(&C(4, 0), c4);
    _mm512_storeu_pd(&C(5, 0), c5);
    _mm512_storeu_pd(&C(6, 0), c6);
    _mm512_storeu_pd(&C(7, 0), c7);
}

void kernel_4x4_4x1(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
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

void kernel_4x4_4x2(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
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

void kernel_4x4_4x3(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
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

void kernel_4x1_1x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
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

void kernel_4x2_2x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
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

void kernel_4x3_3x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
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

void kernel_4x1_1x1(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c10 = C(1, 0), c20 = C(2, 0), c30 = C(3, 0);
    double b00 = B(0, 0);

    c00 += A(0, 0) * b00;
    c10 += A(1, 0) * b00;
    c20 += A(2, 0) * b00;
    c30 += A(3, 0) * b00;

    C(0, 0) = c00;
    C(1, 0) = c10;
    C(2, 0) = c20;
    C(3, 0) = c30;
}

void kernel_4x2_2x2(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1);
    double c10 = C(1, 0), c11 = C(1, 1);
    double c20 = C(2, 0), c21 = C(2, 1);
    double c30 = C(3, 0), c31 = C(3, 1);

    for (int i = 0; i < 2; ++i)
    {
        c00 += A(0, i) * B(i, 0);
        c01 += A(0, i) * B(i, 1);
    }

    for (int i = 0; i < 2; ++i)
    {
        c10 += A(1, i) * B(i, 0);
        c11 += A(1, i) * B(i, 1);
    }

    for (int i = 0; i < 2; ++i)
    {
        c20 += A(2, i) * B(i, 0);
        c21 += A(2, i) * B(i, 1);
    }

    for (int i = 0; i < 2; ++i)
    {
        c30 += A(3, i) * B(i, 0);
        c31 += A(3, i) * B(i, 1);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;

    C(1, 0) = c10;
    C(1, 1) = c11;

    C(2, 0) = c20;
    C(2, 1) = c21;

    C(3, 0) = c30;
    C(3, 1) = c31;
}

void kernel_4x3_3x3(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2);
    double c20 = C(2, 0), c21 = C(2, 1), c22 = C(2, 2);
    double c30 = C(3, 0), c31 = C(3, 1), c32 = C(3, 2);

    for (int i = 0; i < 3; ++i)
    {
        c00 += A(0, i) * B(i, 0);
        c01 += A(0, i) * B(i, 1);
        c02 += A(0, i) * B(i, 2);
    }

    for (int i = 0; i < 3; ++i)
    {
        c10 += A(1, i) * B(i, 0);
        c11 += A(1, i) * B(i, 1);
        c12 += A(1, i) * B(i, 2);
    }

    for (int i = 0; i < 3; ++i)
    {
        c20 += A(2, i) * B(i, 0);
        c21 += A(2, i) * B(i, 1);
        c22 += A(2, i) * B(i, 2);
    }

    for (int i = 0; i < 3; ++i)
    {
        c30 += A(3, i) * B(i, 0);
        c31 += A(3, i) * B(i, 1);
        c32 += A(3, i) * B(i, 2);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;

    C(1, 0) = c10;
    C(1, 1) = c11;
    C(1, 2) = c12;

    C(2, 0) = c20;
    C(2, 1) = c21;
    C(2, 2) = c22;

    C(3, 0) = c30;
    C(3, 1) = c31;
    C(3, 2) = c32;
}

void kernel_1x4_4x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);

    for (int i = 0; i < 4; ++i)
    {
        c00 += A(0, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c01 += A(0, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c02 += A(0, i) * B(i, 2);
    }

    for (int i = 0; i < 4; ++i)
    {
        c03 += A(0, i) * B(i, 3);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;
    C(0, 3) = c03;
}

void kernel_2x4_4x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2), c13 = C(1, 3);

    for (int i = 0; i < 4; ++i)
    {
        c00 += A(0, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c01 += A(0, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c02 += A(0, i) * B(i, 2);
    }

    for (int i = 0; i < 4; ++i)
    {
        c03 += A(0, i) * B(i, 3);
    }

    for (int i = 0; i < 4; ++i)
    {
        c10 += A(1, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c11 += A(1, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c12 += A(1, i) * B(i, 2);
    }

    for (int i = 0; i < 4; ++i)
    {
        c13 += A(1, i) * B(i, 3);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;
    C(0, 3) = c03;

    C(1, 0) = c10;
    C(1, 1) = c11;
    C(1, 2) = c12;
    C(1, 3) = c13;
}

void kernel_3x4_4x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2), c13 = C(1, 3);
    double c20 = C(2, 0), c21 = C(2, 1), c22 = C(2, 2), c23 = C(2, 3);

    for (int i = 0; i < 4; ++i)
    {
        c00 += A(0, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c01 += A(0, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c02 += A(0, i) * B(i, 2);
    }

    for (int i = 0; i < 4; ++i)
    {
        c03 += A(0, i) * B(i, 3);
    }

    for (int i = 0; i < 4; ++i)
    {
        c10 += A(1, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c11 += A(1, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c12 += A(1, i) * B(i, 2);
    }

    for (int i = 0; i < 4; ++i)
    {
        c13 += A(1, i) * B(i, 3);
    }

    for (int i = 0; i < 4; ++i)
    {
        c20 += A(2, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c21 += A(2, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c22 += A(2, i) * B(i, 2);
    }

    for (int i = 0; i < 4; ++i)
    {
        c23 += A(2, i) * B(i, 3);
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
}

void kernel_1x4_4x1(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c0 = C(0, 0);

    for (int i = 0; i < 4; ++i)
    {
        c0 += A(0, i) * B(i, 0);
    }

    C(0, 0) = c0;
}

void kernel_2x4_4x2(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1);
    double c10 = C(1, 0), c11 = C(1, 1);

    for (int i = 0; i < 4; ++i)
    {
        c00 += A(0, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c01 += A(0, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c10 += A(1, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c11 += A(1, i) * B(i, 1);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;

    C(1, 0) = c10;
    C(1, 1) = c11;
}

void kernel_3x4_4x3(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2);
    double c20 = C(2, 0), c21 = C(2, 1), c22 = C(2, 2);

    for (int i = 0; i < 4; ++i)
    {
        c00 += A(0, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c01 += A(0, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c02 += A(0, i) * B(i, 2);
    }

    for (int i = 0; i < 4; ++i)
    {
        c10 += A(1, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c11 += A(1, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c12 += A(1, i) * B(i, 2);
    }

    for (int i = 0; i < 4; ++i)
    {
        c20 += A(2, i) * B(i, 0);
    }

    for (int i = 0; i < 4; ++i)
    {
        c21 += A(2, i) * B(i, 1);
    }

    for (int i = 0; i < 4; ++i)
    {
        c22 += A(2, i) * B(i, 2);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;

    C(1, 0) = c10;
    C(1, 1) = c11;
    C(1, 2) = c12;

    C(2, 0) = c20;
    C(2, 1) = c21;
    C(2, 2) = c22;
}

void kernel_1x1_1x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);
    double a00 = A(0, 0);

    c00 += a00 * B(0, 0);
    c01 += a00 * B(0, 1);
    c02 += a00 * B(0, 2);
    c03 += a00 * B(0, 3);

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;
    C(0, 3) = c03;
}

void kernel_2x2_2x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2), c13 = C(1, 3);

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

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;
    C(0, 3) = c03;

    C(1, 0) = c10;
    C(1, 1) = c11;
    C(1, 2) = c12;
    C(1, 3) = c13;
}

void kernel_3x3_3x4(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2), c03 = C(0, 3);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2), c13 = C(1, 3);
    double c20 = C(2, 0), c21 = C(2, 1), c22 = C(2, 2), c23 = C(2, 3);

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
}

void kernel_1x1_1x1(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    C(0, 0) += A(0, 0) * B(0, 0);
}

void kernel_2x2_2x2(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1);
    double c10 = C(1, 0), c11 = C(1, 1);

    for (int i = 0; i < 2; ++i)
    {
        c00 += A(0, i) * B(i, 0);
        c01 += A(0, i) * B(i, 1);
    }

    for (int i = 0; i < 2; ++i)
    {
        c10 += A(1, i) * B(i, 0);
        c11 += A(1, i) * B(i, 1);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;

    C(1, 0) = c10;
    C(1, 1) = c11;
}

void kernel_3x3_3x3(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    double c00 = C(0, 0), c01 = C(0, 1), c02 = C(0, 2);
    double c10 = C(1, 0), c11 = C(1, 1), c12 = C(1, 2);
    double c20 = C(2, 0), c21 = C(2, 1), c22 = C(2, 2);

    for (int i = 0; i < 3; ++i)
    {
        c00 += A(0, i) * B(i, 0);
        c01 += A(0, i) * B(i, 1);
        c02 += A(0, i) * B(i, 2);
    }

    for (int i = 0; i < 3; ++i)
    {
        c10 += A(1, i) * B(i, 0);
        c11 += A(1, i) * B(i, 1);
        c12 += A(1, i) * B(i, 2);
    }

    for (int i = 0; i < 3; ++i)
    {
        c20 += A(2, i) * B(i, 0);
        c21 += A(2, i) * B(i, 1);
        c22 += A(2, i) * B(i, 2);
    }

    C(0, 0) = c00;
    C(0, 1) = c01;
    C(0, 2) = c02;

    C(1, 0) = c10;
    C(1, 1) = c11;
    C(1, 2) = c12;

    C(2, 0) = c20;
    C(2, 1) = c21;
    C(2, 2) = c22;
}

// end of kernels

// start of ears
void ear_4x4_4x4n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        kernel_4x4_4x4(Ma, Mb, Mc, a, &B(0, i * 4), &C(0, i * 4));
    }
}

void ear_4x4n_4nxk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    if (k == 1)
    {
        for (int i = 0; i < N; ++i)
        {
            kernel_4x4_4x1(Ma, Mb, Mc, &A(0, i * 4), &B(i * 4, 0), c);
        }
    }
    else if (k == 2)
    {
        for (int i = 0; i < N; ++i)
        {
            kernel_4x4_4x2(Ma, Mb, Mc, &A(0, i * 4), &B(i * 4, 0), c);
        }
    }
    else if (k == 3)
    {
        for (int i = 0; i < N; ++i)
        {
            kernel_4x4_4x3(Ma, Mb, Mc, &A(0, i * 4), &B(i * 4, 0), c);
        }
    }
    else
    {
        // should never reach here
    }
}

void ear_4xk_kx4n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    if (k == 1)
    {
        for (int i = 0; i < N; ++i)
        {
            kernel_4x1_1x4(Ma, Mb, Mc, a, &B(0, i * 4), &C(0, i * 4));
        }
    }
    else if (k == 2)
    {
        for (int i = 0; i < N; ++i)
        {
            kernel_4x2_2x4(Ma, Mb, Mc, a, &B(0, i * 4), &C(0, i * 4));
        }
    }
    else if (k == 3)
    {
        for (int i = 0; i < N; ++i)
        {
            kernel_4x3_3x4(Ma, Mb, Mc, a, &B(0, i * 4), &C(0, i * 4));
        }
    }
    else
    {
        // should never reach here
    }
}

void ear_4xk_kxk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    if (k == 1)
    {
        kernel_4x1_1x1(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 2)
    {
        kernel_4x2_2x2(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 3)
    {
        kernel_4x3_3x3(Ma, Mb, Mc, a, b, c);
    }
    else
    {
        // should never reach here
    }
}

void ear_kx4_4x4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    if (k == 1)
    {
        kernel_1x4_4x4(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 2)
    {
        kernel_2x4_4x4(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 3)
    {
        kernel_3x4_4x4(Ma, Mb, Mc, a, b, c);
    }
    else
    {
        // should never reach here
    }
}

void ear_kx4_4xk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    if (k == 1)
    {
        kernel_1x4_4x1(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 2)
    {
        kernel_2x4_4x2(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 3)
    {
        kernel_3x4_4x3(Ma, Mb, Mc, a, b, c);
    }
    else
    {
        // should never reach here
    }
}

void ear_kxk_kx4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    if (k == 1)
    {
        kernel_1x1_1x4(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 2)
    {
        kernel_2x2_2x4(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 3)
    {

        kernel_3x3_3x4(Ma, Mb, Mc, a, b, c);
    }
    else
    {
        // should never reach here
    }
}

void ear_4x8_8x4(
    const int Ma,
    const int Mb,
    const int Mc,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < 2; ++i)
    {
        kernel_4x4_4x4(Ma, Mb, Mc, &A(0, i * 4), &B(i * 4, 0), c);
    }
}

void ear_8x8_8x8n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        kernel_8x8_8x8(Ma, Mb, Mc, a, &B(0, i * 8), &C(0, i * 8));
    }
}

void ear_4x4_4x8(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        kernel_4x4_4x4(Ma, Mb, Mc, a, &B(0, i * 4), &C(0, i * 4));
    }
}

void ear_8x4_4x4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < 2; ++i)
    {
        kernel_4x4_4x4(Ma, Mb, Mc, &A(i * 4, 0), b, &C(i * 4, 0));
    }
}

// start of fields
/*
    No need to provide information about matrix shape because we assume that
    A can be composed of N blocks in a row, where each block is 4 by 4
*/
void field_4x4n_4nx4n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_4x4_4x4n(Ma, Mb, Mc, N, &A(0, i * 4), &B(i * 4, 0), c);
    }
}

void field_kx4_4x4n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_kx4_4x4(Ma, Mb, Mc, k, a, &B(0, i * 4), &C(0, i * 4));
    }
}

void field_8x8_8x4(
    const int Ma,
    const int Mb,
    const int Mc,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < 2; ++i)
    {
        ear_4x8_8x4(Ma, Mb, Mc, &A(i * 4, 0), b, &C(i * 4, 0));
    }
}

void field_8x8n_8nx8n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_8x8_8x8n(Ma, Mb, Mc, N, &A(0, i * 8), &B(i * 8, 0), c);
    }
}

void field_8x8n_8nx4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        field_8x8_8x4(Ma, Mb, Mc, &A(0, i * 8), &B(i * 8, 0), c);
    }
}

void field_4x4_4x8n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_4x4_4x8(Ma, Mb, Mc, N, a, &B(0, i * 8), &C(0, i * 8));
    }
}

void field_8x4_4x8n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < 2; ++i)
    {
        field_4x4_4x8n(Ma, Mb, Mc, N, &A(i * 4, 0), b, &C(i * 4, 0));
    }
}

void field_4x8_8x8n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < 2; ++i)
    {
        field_4x4_4x8n(Ma, Mb, Mc, N, &A(0, i * 4), &B(i * 4, 0), c);
    }
}

// start of farms
void farm_4nx4n_4nx4n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{

    for (int i = 0; i < N; ++i)
    {
        field_4x4n_4nx4n(Ma, Mb, Mc, N, &A(i * 4, 0), b, &C(i * 4, 0));
    }
}

void farm_4nx4n_4nxk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_4x4n_4nxk(Ma, Mb, Mc, N, k, &A(i * 4, 0), b, &C(i * 4, 0));
    }
}

void farm_4nxk_kx4n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_4xk_kx4n(Ma, Mb, Mc, k, N, &A(i * 4, 0), b, &C(i * 4, 0));
    }
}

void farm_4nxk_kxk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_4xk_kxk(Ma, Mb, Mc, k, &A(i * 4, 0), b, &C(i * 4, 0));
    }
}

void farm_kx4n_4nx4n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        field_kx4_4x4n(Ma, Mb, Mc, k, N, &A(0, i * 4), &B(i * 4, 0), c);
    }
}

void farm_kx4n_4nxk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_kx4_4xk(Ma, Mb, Mc, k, &A(0, i * 4), &B(i * 4, 0), c);
    }
}

void farm_kxk_kx4n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_kxk_kx4(Ma, Mb, Mc, k, a, &B(0, i * 4), &C(0, i * 4));
    }
}

void farm_kxk_kxk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    if (k == 1)
    {
        kernel_1x1_1x1(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 2)
    {
        kernel_2x2_2x2(Ma, Mb, Mc, a, b, c);
    }
    else if (k == 3)
    {
        kernel_3x3_3x3(Ma, Mb, Mc, a, b, c);
    }
    else
    {
        // should never reach here
    }
}

void farm_8nx8n_8nx4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        field_8x8n_8nx4(Ma, Mb, Mc, N, &A(i * 8, 0), b, &C(i * 8, 0));
    }
}

void farm_8nx8n_8nx8n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        field_8x8n_8nx8n(Ma, Mb, Mc, N, &A(i * 8, 0), b, &C(i * 8, 0));
    }
}

void farm_8nx4_4x8n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        field_8x4_4x8n(Ma, Mb, Mc, N, &A(i * 8, 0), b, &C(i * 8, 0));
    }
}

void farm_8nx4_4x4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_8x4_4x4(Ma, Mb, Mc, N, &A(i * 8, 0), b, &C(i * 8, 0));
    }
}

void farm_4x8n_8nx8n(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        field_4x8_8x8n(Ma, Mb, Mc, N, &A(0, i * 8), &B(i * 8, 0), c);
    }
}

void farm_4x8n_8nx4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    for (int i = 0; i < N; ++i)
    {
        ear_4x8_8x4(Ma, Mb, Mc, &A(0, i * 8), &B(i * 8, 0), c);
    }
}

// ranches start here

// 4n x 4n multiplied by 4n x (4n + k)
void ranch_4nx4n_4nx4nk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    // 4n x 4n multiplied by 4n x (4n)
    farm_4nx4n_4nx4n(Ma, Mb, Mc, N, a, b, c);

    // 4n x 4n multiplied by 4n x k
    farm_4nx4n_4nxk(Ma, Mb, Mc, N, k, a, &B(0, N * 4), &C(0, N * 4));
}

// 4n x k multiplied by k x (4n + k)
void ranch_4nxk_kx4nk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    const int k,
    double *restrict a,
    double *restrict b,
    double *c)
{
    // 4n x k multiplied by k x 4n
    farm_4nxk_kx4n(Ma, Mb, Mc, N, k, a, b, c);

    // 4n x k multiplied by k x k
    farm_4nxk_kxk(Ma, Mb, Mc, N, k, a, &B(0, N * 4), &C(0, N * 4));
}

// k x 4n multiplied by 4n x (4n + k)
void ranch_kx4n_4nx4nk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    // k x 4n multiplied by 4n x 4n
    farm_kx4n_4nx4n(Ma, Mb, Mc, k, N, a, b, c);

    // k x 4n multiplied by 4n x k
    farm_kx4n_4nxk(Ma, Mb, Mc, k, N, a, &B(0, N * 4), &C(0, N * 4));
}

// k x k multiplied by k x (4n + k)
void ranch_kxk_kx4nk(
    const int Ma,
    const int Mb,
    const int Mc,
    const int k,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    // k x k multiplied by k x 4n
    farm_kxk_kx4n(Ma, Mb, Mc, k, N, a, b, c);

    // k x k multiplied by k x k
    farm_kxk_kxk(Ma, Mb, Mc, k, a, &B(0, N * 4), &C(0, N * 4));
}

void ranch_8nx8n_8nx8n4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    const int plus4,
    double *restrict a,
    double *restrict b,
    double *c)
{
    // 8nx8n multiplied by 8nx8n
    farm_8nx8n_8nx8n(Ma, Mb, Mc, N, a, b, c);

    // 8nx8n multiplied by 8nx4
    if (plus4 > 0)
    {
        farm_8nx8n_8nx4(Ma, Mb, Mc, N, a, &B(0, N * 8), &C(0, N * 8));
    }
}

void ranch_8nx4_4x8n4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    // 8nx4 multiplied by 4x8n
    farm_8nx4_4x8n(Ma, Mb, Mc, N, a, b, c);

    // 8nx4 multiplied by 4x4
    farm_8nx4_4x4(Ma, Mb, Mc, N, a, &B(0, N * 8), &C(0, N * 8));
}

void ranch_4x8n_8nx8n4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    // 4x8n multiplied by 8nx8n
    farm_4x8n_8nx8n(Ma, Mb, Mc, N, a, b, c);

    // 4x8n multiplied by 8nx4
    farm_4x8n_8nx4(Ma, Mb, Mc, N, a, &B(0, N * 8), &C(0, N * 8));
}

void ranch_4x4_4x8n4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    // 4x4 multiplied by 4x8n
    field_4x4_4x8n(Ma, Mb, Mc, N, a, b, c);

    // 4x4 multiplied by 4x4
    kernel_4x4_4x4(Ma, Mb, Mc, a, &B(0, N * 8), &C(0, N * 8));
}

const char *dgemm_desc = "My dgemm.";

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

void monsanto(const int M, double *restrict a, double *restrict b, double *restrict c)
{
    const int N = M / 4;
    const int k = M % 4;
    const int Ma = M;
    const int Mb = M;
    const int Mc = M;
    ranch_4nx4n_4nx4nk(Ma, Mb, Mc, N, k, a, b, c);
    ranch_4nxk_kx4nk(Ma, Mb, Mc, N, k, &A(0, 4 * N), &B(4 * N, 0), c);
    ranch_kx4n_4nx4nk(Ma, Mb, Mc, k, N, &A(4 * N, 0), b, &C(4 * N, 0));
    ranch_kxk_kx4nk(Ma, Mb, Mc, k, N, &A(4 * N, 4 * N), &B(4 * N, 0), &C(4 * N, 0));
}

void square_dgemm(const int M, double *restrict a, double *restrict b, double *c)
{
    monsanto(M, b, a, c);
}