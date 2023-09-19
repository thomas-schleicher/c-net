#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

// operational functions
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
    printf("Rows: %d, Columns: %d\n", matrix->rows, matrix->columns);

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

// mathematical functions

/*
 * These methods won't change or free the input matrix.
 * It creates a new matrix, which is modified and then returned.
 * If we don't need the original matrix, we should consider just changing the original matrix and changing the method signature to void.
 */

Matrix* multiply(Matrix* matrix1, Matrix* matrix2) {

    // check if the two matrices are of the same size
    if(matrix1->rows != matrix2->rows || matrix1->columns != matrix2->columns) {
        printf("ERROR: Size of matrices are not compatible! (Multiply)");
        exit(1);
    }

    // create result matrix
    Matrix* result_matrix = matrix_create(matrix1->rows, matrix1->columns);

    // multiply the values and save them into the result matrix
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix1->columns; j++) {
            result_matrix->numbers[i][j] = matrix1->numbers[i][j] * matrix2->numbers[i][j];
        }
    }

    // return resulting matrix
    return  result_matrix;
}

Matrix* add(Matrix* matrix1, Matrix* matrix2) {

    // check if the two matrices are of the same size
    if(matrix1->rows != matrix2->rows || matrix1->columns != matrix2->columns) {
        printf("ERROR: Size of matrices are not compatible! (Add)");
        exit(1);
    }

    // create result matrix
    Matrix* result_matrix = matrix_create(matrix1->rows, matrix1->columns);

    // add the value of the number in matrix 1 to the value of the number in matrix 2
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix1->columns; j++) {
            result_matrix->numbers[i][j] = matrix1->numbers[i][j] + matrix2->numbers[i][j];
        }
    }

    // return the result matrix
    return result_matrix;
}

Matrix* subtract(Matrix* matrix1, Matrix* matrix2) {

    // check if the two matrices are of the same size
    if(matrix1->rows != matrix2->rows || matrix1->columns != matrix2->columns) {
        printf("ERROR: Size of matrices are not compatible! (Subtract)");
        exit(1);
    }

    // create result matrix
    Matrix* result_matrix = matrix_create(matrix1->rows, matrix1->columns);

    // subtract the value of the number in matrix 2 from the value of the number in matrix 1
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix1->columns; j++) {
            result_matrix->numbers[i][j] = matrix1->numbers[i][j] - matrix2->numbers[i][j];
        }
    }

    // return the resulting matrix
    return result_matrix;
}

Matrix* dot(Matrix* matrix1, Matrix* matrix2) {

    // check if the dimensions of the matrices are compatible to calculate the dot product
    if(matrix1->columns != matrix2->rows) {
        printf("ERROR: Size of matrices are not compatible! (Dot-Product)");
        exit(1);
    }

    // create a new matrix with the dimensions of the dot product;
    Matrix* result_matrix = matrix_create(matrix1->rows, matrix2->columns);

    // iterate through all rows of matrix 1
    for (int i = 0; i < matrix1->rows; i++) {

        // iterate though all columns of matrix 2
        for (int j = 0; j < matrix2->columns; j++) {

            // sum up the products and save them into the result matrix
            result_matrix->numbers[i][j] = 0;
            for (int k = 0; k < matrix2->rows; k++) {
                result_matrix->numbers[i][j] += matrix1->numbers[i][k] * matrix2->numbers[k][j];
            }
        }
    }

    // return result
    return result_matrix;
}

Matrix* apply(double (*function)(double), Matrix* matrix) {

    // create a new matrix used to calculate the result
    Matrix* result_matrix = matrix_create(matrix->rows, matrix->columns);

    // apply the function to all values in the matrix
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            matrix->numbers[i][j] = (*function)(matrix->numbers[i][j]);
        }
    }

    // return resulting matrix
    return result_matrix;
}

Matrix* scale(Matrix* matrix, double value) {

    // create a copy of the original matrix
    Matrix* result_matrix = matrix_copy(matrix);

    // iterate over all numbers in the matrix and multiply by the scalar value
    for (int i = 0; i < result_matrix->rows; i++) {
        for (int j = 0; j < result_matrix->columns; j++) {
            result_matrix->numbers[i][j] *= value;
        }
    }

    // return the copy
    return result_matrix;
}

Matrix* addScalar(Matrix* matrix, double value) {

    // create a copy of the original matrix
    Matrix* result_matrix = matrix_copy(matrix);

    // iterate over all numbers in the matrix and add the scalar value
    for (int i = 0; i < result_matrix->rows; i++) {
        for (int j = 0; j < result_matrix->columns; j++) {
            result_matrix->numbers[i][j] += value;
        }
    }

    // return the copy
    return result_matrix;
}

Matrix* transpose(Matrix* matrix) {

    // create a new matrix of the size n-m, based on the original matrix of size m-n
    Matrix* result_matrix = matrix_create(matrix->columns, matrix->rows);

    // copy the values from the original into the correct place in the copy
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            result_matrix->numbers[j][i] = matrix->numbers[i][j];
        }
    }

    // return the result matrix
    return result_matrix;

}

//file operations
void matrix_save(Matrix* matrix, char* file_string){
    FILE *fptr = fopen(file_string, "w+");
    if(!fptr){
        printf("Unable to get handle for \"%s\"", file_string);
        exit(1);
    }
    for(int i = 0; i < matrix->rows; i++){
        for(int j = 0; j < matrix->columns; j++){
            fprintf(fptr, "%f.12 ", matrix->numbers[i][j]);
        }
        fputc('\n', fptr);
    }
}


Matrix* matrix_flatten(Matrix* matrix, int axis) {
    // Axis = 0 -> Column Vector, Axis = 1 -> Row Vector
    Matrix* result_matrix;
    // Column Vector
    if (axis == 0) {
        result_matrix = matrix_create(matrix -> rows * matrix -> columns, 1);
    }
    // Row Vector
    else if (axis == 1) {
        result_matrix = matrix_create(1, matrix -> rows * matrix -> columns);
    } else {
        printf("ERROR: Argument must be 1 or 0 (matrix_flatten");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            if (axis == 0) result_matrix->numbers[i * matrix->columns + j][0] = matrix->numbers[i][j];
            else if (axis == 1) result_matrix->numbers[0][i * matrix->columns + j] = matrix->numbers[i][j];
        }
    }
    return result_matrix;
}
