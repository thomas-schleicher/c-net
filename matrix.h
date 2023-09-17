#pragma once

typedef struct {
    int rows, columns;
    double **numbers;
} Matrix;

// operational functions
Matrix* matrix_create(int rows, int columns);
void matrix_fill(Matrix* matrix, double value);
void matrix_free(Matrix* matrix);
void matrix_print(Matrix *matrix);
Matrix* matrix_copy(Matrix *matrix);

// mathematical functions
Matrix* multiply(Matrix* matrix1, Matrix* matrix2);
Matrix* add(Matrix* matrix1, Matrix* matrix2);
Matrix* subtract(Matrix* matrix1, Matrix* matrix2);
Matrix* dot(Matrix* matrix1, Matrix* matrix2);
Matrix* apply(double (*function)(double), Matrix* matrix);
Matrix* scale(Matrix* matrix, double value);
Matrix* addScalar(Matrix* matrix, double value);
Matrix* transpose(Matrix* matrix);



