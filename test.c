#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function prototype for square_dgemm
void square_dgemm(const int M, const double *A, const double *B, double *C);

// Function prototype for transpose
double *transpose(const int N, const double *X);

void square_dgemm_base(const int M, const double *A, const double *B, double *C)
{
    int i, j, k;
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

// Function to check for the first difference between two matrices
void check_matrix_difference(const double *C, const double *C_base, int M, double tolerance)
{
    for (int i = 0; i < M * M; i++)
    {
        if (fabs(C[i] - C_base[i]) > tolerance)
        {
            int row = i / M;
            int col = i % M;
            printf("First difference found at position (%d, %d):\n", row, col);
            printf("C[%d, %d] = %f\n", row, col, C[i]);
            printf("C_base[%d, %d] = %f\n", row, col, C_base[i]);
            printf("Absolute difference: %f\n", fabs(C[i] - C_base[i]));
            return;
        }
    }
    printf("No differences found within tolerance of %e\n", tolerance);
}

// Function to print a matrix
void print_matrix(const char *name, const double *matrix, int rows, int cols)
{
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%7.5f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void initialize_random_matrix(double *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        matrix[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random double between -1 and 1
    }
}

int main()
{
    const int M = 4; // M by M matrices

    // Allocate memory for matrices
    double *A = (double *)malloc(M * M * sizeof(double));
    double *B = (double *)malloc(M * M * sizeof(double));
    double *C = (double *)calloc(M * M, sizeof(double));      // Initialize C with zeros
    double *C_base = (double *)calloc(M * M, sizeof(double)); // Initialize C with zeros

    // Initialize matrix A (row-major order)
    initialize_random_matrix(A, M * M);

    // Initialize matrix B (row-major order)
    initialize_random_matrix(B, M * M);

    // Print initial matrices
    print_matrix("Matrix A (row-major)", A, M, M);
    print_matrix("Matrix B (row-major)", B, M, M);

    // Perform matrix multiplication
    square_dgemm(M, A, B, C);

    // Perform matrix multiplication reference
    square_dgemm_base(M, A, B, C_base);

    // Print result
    print_matrix("Result Matrix C", C, M, M);

    // Print result reference
    print_matrix("Result Matrix C (Reference)", C_base, M, M);

    const double tolerance = 1e-10;

    // Check for differences
    check_matrix_difference(C, C_base, M, tolerance);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}