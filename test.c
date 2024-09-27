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

// simulated 8x8_8x8 kernel since we don't have avx512 yet
void simulated_kernel_8x8_8x8(const int Ma, const int Mb, const int Mc, double *restrict a, double *restrict b, double *c)
{
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                kernel_4x4_4x4(Ma, Mb, Mc, &A(i * 4, k * 4), &B(k * 4, j * 4), &C(i * 4, j * 4));
            }
        }
    }
}

// start of ears
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
        simulated_kernel_8x8_8x8(Ma, Mb, Mc, a, &B(0, i * 8), &C(0, i * 8));
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

// start of ranches

// 8nx8n multiplied by 8nx(8n + 4) matrix
void ranch_8nx8n_8nx8n4(
    const int Ma,
    const int Mb,
    const int Mc,
    const int N,
    double *restrict a,
    double *restrict b,
    double *c)
{
    // 8nx8n multiplied by 8nx8n
    farm_8nx8n_8nx8n(Ma, Mb, Mc, N, a, b, c);

    // 8nx8n multiplied by 8nx4
    farm_8nx8n_8nx4(Ma, Mb, Mc, N, a, &B(0, N * 8), &C(0, N * 8));
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
    const int M = 8;
    const int N = 2;
    const int k = 4;
    const int Ma = M * N + k;
    const int Mb = Ma;
    const int Mc = Ma;
    double *a = (double *)malloc(Ma * Ma * sizeof(double));
    double *b = (double *)malloc(Mb * Mb * sizeof(double));
    double *c = (double *)calloc(Mc * Mc, sizeof(double));

    // fill A
    for (int i = 0; i < Ma * Ma; ++i)
    {
        a[i] = i / 10.0;
    }
    // fill B
    for (int i = 0; i < Mb * Mb; ++i)
    {
        b[i] = (i) * -1 / 10.0;
    }

    // print out what A and B look like
    printf("A:\n");
    for (int i = 0; i < Ma * Ma; ++i)
    {
        printf("%.2f ", a[i]);
        if ((i + 1) % Ma == 0)
        {
            printf("\n");
        }
    }
    // and B
    printf("B:\n");
    for (int i = 0; i < Mb * Mb; ++i)
    {
        printf("%.2f ", b[i]);
        if ((i + 1) % (Mb) == 0)
        {
            printf("\n");
        }
    }

    // ranch_8nx8n_8nx8n4(Ma, Mb, Mc, N, a, b, c);
    // ranch_8nx4_4x8n4(Ma, Mb, Mc, N, &A(0, N * 8), &B(N * 8, 0), c);
    farm_4x8n_8nx8n(Ma, Mb, Mc, N, a, b, c);

    // print result
    printf("Result:\n");
    for (int i = 0; i < Mc; ++i)
    {
        for (int j = 0; j < Mc; ++j)
        {
            printf("%.2f ", C(i, j));
        }
        printf("\n");
    }
}