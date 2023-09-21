#pragma once

#include "matrix.h"
#include "image.h"

typedef struct {
    int input_size;
    //Matrix* input; as local variable given to function

    // hidden layers
    int hidden_size;
    Matrix* weights_1;
    Matrix* bias_1;
    Matrix* weights_2;
    Matrix* bias_2;
    Matrix* weights_3;
    Matrix* bias_3;

    int output_size;
    Matrix* weights_output;
    //Matrix* bias_output; // do we need it?
    //Matrix* output; as local variable given to function

    double learning_rate;

} Neural_Network;

static const int MAX_BYTES = 100;

Neural_Network* new_network(int input_size, int hidden_size, int output_size, double learning_rate);
//void print_network(Neural_Network* network);
void free_network(Neural_Network* network);

void save_network(Neural_Network* network);
Neural_Network* load_network(char* file);

double predict_images(Neural_Network* network, Image** images, int amount);
Matrix* predict_image(Neural_Network* network, Image* image);
Matrix* predict(Neural_Network* network, Matrix* image_data);

void train_network(Neural_Network* network, Matrix* input, Matrix* output);
void batch_train_network(Neural_Network* network, Image** images, int size);

double relu(double input);
Matrix* softmax(Matrix* matrix);
