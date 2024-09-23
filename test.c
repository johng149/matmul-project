#include <stdio.h>
#include <stdlib.h>

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

// start of farms
void farm_4nx4n_4nx4n(
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    const int Ma = 4 * N;
    const int Mb = Ma;
    const int Mc = Ma;
    for (int i = 0; i < N; ++i)
    {
        field_4x4n_4nx4n(Ma, Mb, Mc, N, &A(i * 4, 0), b, &C(i * 4, 0));
    }
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

void square_dgemm(const int Ma, const int Mb, const int Mc, double *a, double *b, double *c)
{
    kernel_4x4_4x4(Ma, Mb, Mc, a, b, c);
}

void main()
{
    const int M = 4;
    const int N = 2;
    const int k = 3;
    const int Ma = M * N;
    const int Mb = M * N;
    const int Mc = M * N;
    double *a = (double *)malloc(M * N * M * N * sizeof(double));
    double *b = (double *)malloc(M * N * M * N * sizeof(double));
    double *c = (double *)calloc(M * N * M * N, sizeof(double));

    // fill A
    for (int i = 0; i < M * N * M * N; ++i)
    {
        a[i] = i;
    }
    // fill B
    for (int i = 0; i < M * N * M * N; ++i)
    {
        b[i] = i * -1;
    }

    // print out what A and B look like
    printf("A:\n");
    for (int i = 0; i < M * M * N; ++i)
    {
        printf("%f ", a[i]);
        if ((i + 1) % (M * N) == 0)
        {
            printf("\n");
        }
    }
    // and B
    printf("B:\n");
    for (int i = 0; i < M * N * M * N; ++i)
    {
        printf("%f ", b[i]);
        if ((i + 1) % (M * N) == 0)
        {
            printf("\n");
        }
    }

    ear_4x4n_4nxk(Ma, Mb, Mc, N, k, a, b, c);

    // print result
    printf("Result:\n");
    for (int i = 0; i < M * N; ++i)
    {
        for (int j = 0; j < M * N; ++j)
        {
            printf("%f ", C(i, j));
        }
        printf("\n");
    }
}