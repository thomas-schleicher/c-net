#include <stdlib.h>
#include "neuronal_network.h"

Neural_Network* new_network(int input_size, int hidden_size, int output_size, double learning_rate){
    Neural_Network network = malloc(sizeof(Neural_Network));
    // initialize networks variables
    network.input_size = input_size;
    network.hidden_size = hidden_size;
    network.output_size = output_size;
    network.learning_rate = learning_rate;

    network.weights_1 = matrix_randomize(matrix_create(hidden_size, input_size));
    network.weights_2 = matrix_randomize(matrix_create(hidden_size, hidden_size));
    network.weights_3 = matrix_randomize(matrix_create(hidden_size, hidden_size));
    network.weights_output = matrix_randomize(matrix_create(output_size, hidden_size));
    network.bias_1 = matrix_randomize(matrix_create(hidden_size, 1));
    network.bias_2 = matrix_randomize(matrix_create(hidden_size, 1));
    network.bias_3 = matrix_randomize(matrix_create(hidden_size, 1));
    //network.bias_output = matrix_create(output_size, 1); // do we need it?

    return network;
}

//void print_network(Neural_Network* network){};

void free_network(Neural_Network* network){
    matrix_free(network->weights_1);
    matrix_free(network->weights_2);
    matrix_free(network->weights_3);
    matrix_free(network->weights_output);
    matrix_free(network->bias_1);
    matrix_free(network->bias_2);
    matrix_free(network->bias_3);
    free(network);
}


void save_network(Neural_Network* network, char* file);
Neural_Network* load_network(char* file);

double predict_images(Neural_Network* network, Image** images, int amount);
Matrix* predict_image(Neural_Network* network, Image*);
Matrix* predict(Neural_Network* network, Matrix* image_data);

void train_network(Neural_Network* network, Matrix* input, Matrix* output);
void batch_train_network(Neural_Network* network, Image** images, int size);