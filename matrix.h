#pragma once
#include <stdio.h>

typedef struct {
    int rows, columns;
    double **numbers;
} Matrix;

static const int scaling_value = 10000;

// operational functions
Matrix* matrix_create(int rows, int columns);
void matrix_fill(Matrix* matrix, double value);
void matrix_free(Matrix* matrix);
void matrix_print(Matrix *matrix);
Matrix* matrix_copy(Matrix *matrix);
void matrix_save(Matrix* matrix, char* file_string);
Matrix* matrix_load(char* file_string);
Matrix* load_next_matrix(FILE * save_file);

void matrix_randomize(Matrix* matrix, int n); // don't understand the usage of the n
int matrix_argmax(Matrix* matrix);
Matrix* matrix_flatten(Matrix* matrix, int axis);
Matrix* matrix_add_bias(Matrix* matrix);

/*
 * These methods won't change or free the input matrix.
 * It creates a new matrix, which is modified and then returned.
 * If we don't need the original matrix, we should consider just changing the original matrix and changing the method signature to void.
 */

// mathematical functions
Matrix* multiply(Matrix* matrix1, Matrix* matrix2);
Matrix* add(Matrix* matrix1, Matrix* matrix2);
Matrix* subtract(Matrix* matrix1, Matrix* matrix2);
Matrix* dot(Matrix* matrix1, Matrix* matrix2);
Matrix* apply(double (*function)(double), Matrix* matrix);
Matrix* scale(Matrix* matrix, double value);
Matrix* transpose(Matrix* matrix);
