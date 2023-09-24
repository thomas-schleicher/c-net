#pragma once
#include <stdio.h>

typedef struct {
    int rows, columns;
    double **numbers;
} Matrix;

static const int scaling_value = 10000;

Matrix* matrix_create(int rows, int columns);
void matrix_fill(Matrix* matrix, double value);
void matrix_free(Matrix* matrix);
void matrix_print(Matrix *matrix);
Matrix* matrix_copy(Matrix *matrix);
void matrix_save(Matrix* matrix, char* file_string);
Matrix* matrix_load(char* file_string);
Matrix* load_next_matrix(FILE * save_file);

void matrix_randomize(Matrix* matrix, int n);
int matrix_argmax(Matrix* matrix);
Matrix* matrix_flatten(Matrix* matrix, int axis);
Matrix* matrix_add_bias(Matrix* matrix);

Matrix* multiply(Matrix* matrix1, Matrix* matrix2);
Matrix* add(Matrix* matrix1, Matrix* matrix2);
Matrix* subtract(Matrix* matrix1, Matrix* matrix2);
Matrix* dot(Matrix* matrix1, Matrix* matrix2);
Matrix* apply(double (*function)(double), Matrix* matrix);
Matrix* scale(Matrix* matrix, double value);
Matrix* addScalar(Matrix* matrix, double value);
Matrix* transpose(Matrix* matrix);