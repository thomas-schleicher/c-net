#pragma once

typedef struct {
    int rows, columns;
    double **numbers;
} Matrix;

Matrix* matrix_create(int rows, int columns);
void matrix_fill(Matrix* matrix, double value);
void matrix_free(Matrix* matrix);
void matrix_print(Matrix *matrix);
Matrix* matrix_copy(Matrix *matrix);



