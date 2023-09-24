#include "matrix.h"

static const int scaling_value = 10000;

Matrix* multiply(Matrix* matrix1, Matrix* matrix2);

Matrix* add(Matrix* matrix1, Matrix* matrix2); //only used in the batch_training method

Matrix* subtract(Matrix* matrix1, Matrix* matrix2);

Matrix* dot(Matrix* matrix1, Matrix* matrix2);

Matrix* apply(double (*function)(double), Matrix* matrix);

Matrix* scale(Matrix* matrix, double value);

Matrix* transpose(Matrix* matrix);

Matrix* matrix_flatten(Matrix* matrix, int axis);

int argmax(Matrix* matrix);

void matrix_randomize(Matrix* matrix, int n);

Matrix* matrix_add_bias(Matrix* matrix);