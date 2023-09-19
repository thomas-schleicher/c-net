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

double predict_images(Neural_Network* network, Image** images, int amount) {
    int num_correct = 0;
    for (int i = 0; i < amount; i++) {
        Matrix* prediction = predict_image(network, images[i]);
        if (matrix_argmax(prediction) == images[i]->image_label) {
            num_correct++;
        }
        matrix_free(prediction);
    }
    return 1.0 * num_correct / amount;
}
Matrix* predict_image(Neural_Network* network, Image*);

Matrix* predict(Neural_Network* network, Matrix* image_data) {
    Matrix* hidden1_inputs = dot(network->weights_1, image_data);
    Matrix* hidden1_outputs = apply(relu, hidden1_inputs);

    Matrix* hidden2_inputs = dot(network->weights_2, hidden1_outputs);
    Matrix* hidden2_outputs = apply(relu, hidden2_inputs);


    Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
    Matrix* final_outputs = apply(sigmoid, final_inputs);
    Matrix* result = softmax(final_outputs);

    matrix_free(hidden_inputs);
    matrix_free(hidden_outputs);
    matrix_free(final_inputs);
    matrix_free(final_outputs);

    return result;
}

void train_network(Neural_Network* network, Matrix* input, Matrix* output);
void batch_train_network(Neural_Network* network, Image** images, int size);