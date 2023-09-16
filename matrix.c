#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

Matrix* matrix_create(int rows, int columns) {

    // allocate memory for the matrix
    Matrix* matrix = malloc(sizeof(Matrix));

    // set size variables to the correct size
    matrix->rows = rows;
    matrix->columns = columns;

    // allocate memory for the numbers (2D-Array)
    matrix->numbers = malloc(sizeof(double*) * rows);
    for (int i = 0; i < rows; i++) {
        matrix->numbers[i] = malloc(sizeof(double) * columns);
    }

    // return the pointer to the allocated memory
    return matrix;
}

void matrix_fill(Matrix* matrix, double value) {

    // simple for loop to populate the 2D-array with a value
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            matrix->numbers[i][j] = value;
        }
    }
}

void matrix_free(Matrix* matrix) {

    // de-allocate every column
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->numbers[i]);
    }

    // de-allocate the rows
    free(matrix->numbers);

    // de-allocate the matrix
    free(matrix);
}

void matrix_print(Matrix *matrix) {

    // print the dimensions of the matrix
    printf("Rows: %d, Columns: %d", matrix->rows, matrix->columns);

    // loop through all values and format them into the correct matrix representation
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            printf("%lf ", matrix->numbers[i][j]);
        }
        printf("\n");
    }
}

Matrix* matrix_copy(Matrix *matrix) {

    // create another matrix of the same size
    Matrix* copy_of_matrix = matrix_create(matrix->rows, matrix->columns);

    // copy the values from the original matrix into the copy
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            copy_of_matrix->numbers[i][j] = matrix->numbers[i][j];
        }
    }

    // return the pointer to the copy
    return copy_of_matrix;
}

