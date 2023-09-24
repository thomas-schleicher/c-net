#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

#define MAX_BYTES 100

Matrix* matrix_create(int rows, int columns) {

    // allocate memory for the matrix
    Matrix* matrix = malloc(sizeof(Matrix));

    // set size variables to the correct size
    matrix->rows = rows;
    matrix->columns = columns;

    // allocate memory for the numbers (2D-Array)
    matrix->numbers = malloc(sizeof(double*) * rows);
    for (int i = 0; i < rows; i++) {
        matrix->numbers[i] = calloc(sizeof(double), columns);
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

void matrix_save(Matrix* matrix, char* file_string){

    // open the file in append mode
    FILE *file = fopen(file_string, "a");

    // check if the file could be found
    if(file == NULL) {
        printf("ERROR: Unable to get handle for \"%s\"! (matrix_save)", file_string);
        exit(1);
    }

    // save the size of the matrix
    fprintf(file, "%d\n", matrix->rows);
    fprintf(file, "%d\n", matrix->columns);

    // save all the numbers of the matrix into the file
    for(int i = 0; i < matrix->rows; i++){
        for(int j = 0; j < matrix->columns; j++){
            fprintf(file, "%.10f\n", matrix->numbers[i][j]);
        }
    }

    // close the file
    fclose(file);
}

Matrix* matrix_load(char* file_string){

    FILE *fptr = fopen(file_string, "r");

    if(!fptr){
        printf("Could not open \"%s\"", file_string);
        exit(1);
    }

    Matrix * m = load_next_matrix(fptr);

    fclose(fptr);
    return m;
}

Matrix* load_next_matrix(FILE *save_file){

    char buffer[MAX_BYTES];

    fgets(buffer, MAX_BYTES, save_file);
    int rows = (int)strtol(buffer, NULL, 10);
    fgets(buffer, MAX_BYTES, save_file);
    int cols = (int)strtol(buffer, NULL, 10);

    Matrix *matrix = matrix_create(rows, cols);

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            fgets(buffer, MAX_BYTES, save_file);
            matrix->numbers[i][j] = strtod(buffer, NULL);
        }
    }
    return matrix;
}