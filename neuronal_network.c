#include <stdlib.h>
#include "neuronal_network.h"
#include <stdio.h>
#include <math.h>

double sigmoid(double input);
Matrix* predict(Neural_Network* network, Matrix* image_data);
double square(double input);
Matrix * backPropagation(double learning_rate, Matrix* weights, Matrix* biases, Matrix* current_layer_activation, Matrix* previous_layer_activation, Matrix* sigma_old);

Neural_Network* new_network(int input_size, int hidden_size, int hidden_amount, int output_size, double learning_rate){
    Neural_Network* network = malloc(sizeof(Neural_Network));

    network->input_size = input_size;
    network->hidden_size = hidden_size;
    network->hidden_amount = hidden_amount;
    network->output_size = output_size;
    network->learning_rate = learning_rate;

    Matrix** weights = malloc(sizeof(Matrix*) * (hidden_amount + 1));
    network->weights = weights;

    network->weights[0] = matrix_create(hidden_size, input_size + 1);
    for(int i=1;i<hidden_amount;i++){
        network->weights[i] = matrix_create(hidden_size, hidden_size + 1);
    }
    network->weights[hidden_amount] = matrix_create(output_size, hidden_size + 1);

    return network;
}

void randomize_network(Neural_Network* network, int scope){
    for (int i = 0; i < network->hidden_amount + 1; i++) {
        matrix_randomize(network->weights[i], scope);
    }
}

void free_network(Neural_Network* network){
    for (int i = 0; i < network->hidden_amount + 1; i++) {
        matrix_free(network->weights[i]);
    }
    free(network->weights);
    free(network);
}

void save_network(Neural_Network* network) {

    char* file_name = "../networks/newest_network.txt";

    // create file
    FILE* save_file = fopen(file_name, "w");

    // check if file is successfully opened
    if(save_file == NULL) {
        printf("ERROR: Something went wrong in file creation! (save_network)");
        exit(1);
    }

    // save network size to first line of the file
    fprintf(save_file, "%d\n", network->input_size);
    fprintf(save_file, "%d\n", network->hidden_size);
    fprintf(save_file, "%d\n", network->hidden_amount);
    fprintf(save_file, "%d\n", network->output_size);

    // close the file
    fclose(save_file);

    for (int i = 0; i < network->hidden_amount + 1; ++i) {
        matrix_save(network->weights[i], file_name);
    }

    printf("Network Saved!");
}

Neural_Network* load_network(char* file) {

    // create file pointer and open file
    FILE* save_file = fopen(file, "r");

    // check if file could be opened
    if(save_file == NULL) {
        printf("ERROR: File could not be opened/found! (load_network)");
        exit(1);
    }

    // read & store the information on the size of the network from the save file
    char buffer[MAX_BYTES];
    fgets(buffer, MAX_BYTES, save_file);
    int input_size = (int) strtol(buffer, NULL, 10);
    fgets(buffer, MAX_BYTES, save_file);
    int hidden_size = (int) strtol(buffer, NULL, 10);
    fgets(buffer, MAX_BYTES, save_file);
    int hidden_amount = (int) strtol(buffer, NULL, 10);
    fgets(buffer, MAX_BYTES, save_file);
    int output_size = (int) strtol(buffer, NULL, 10);

    // create a new network to fill with the saved data
    Neural_Network* saved_network = new_network(input_size, hidden_size, hidden_amount, output_size, 0);

    for (int i = 0; i < saved_network->hidden_amount + 1; ++i) {
        saved_network->weights[i] = load_next_matrix(save_file);
    }

    // return saved network
    fclose(save_file);
    return saved_network;
}

void print_network(Neural_Network* network) {
    for (int i = 0; i < network->hidden_amount; ++i) {
        matrix_print(network->weights[i]);
    }
}

double measure_network_accuracy(Neural_Network* network, Image** images, int amount) {
    int num_correct = 0;
    for (int i = 0; i < amount; i++) {
        Matrix* prediction = predict_image(network, images[i]);
        if (matrix_argmax(prediction) == images[i]->label) {
            num_correct++;
        }
        matrix_free(prediction);
    }
    return ((double) num_correct) / amount;
}

Matrix* predict_image(Neural_Network* network, Image* image){
    Matrix* image_data = matrix_flatten(image->pixel_values, 0);
    Matrix* res = predict(network, image_data);
    matrix_free(image_data);
    return res;
}

Matrix* predict(Neural_Network* network, Matrix* image_data) {

    Matrix* input = matrix_add_bias(image_data);

    Matrix* output[network->hidden_amount + 1];
    for (int i = 0; i < network->hidden_amount + 1; ++i) {
        Matrix* neuron_input = dot(network->weights[i], input);
        Matrix* neuron_activation = apply(sigmoid, neuron_input);

        output[i] = neuron_activation;

        matrix_free(neuron_input);
        matrix_free(input);

        input = matrix_add_bias(neuron_activation);
    }

    for (int i = 0; i < network->hidden_amount; ++i) {
        matrix_free(output[i]);
    }

    matrix_free(input);

    return output[network->hidden_amount + 1];
}

void train_network(Neural_Network* network, Image *image, int label) {

    Matrix* image_data = matrix_flatten(image->pixel_values, 0);
    Matrix* input = matrix_add_bias(image_data);

    Matrix* output[network->hidden_amount + 1];
    for (int i = 0; i < network->hidden_amount + 1; ++i) {
        Matrix* neuron_input = dot(network->weights[i], input);
        Matrix* neuron_activation = apply(sigmoid, neuron_input);

        output[i] = neuron_activation;

        matrix_free(neuron_input);
        matrix_free(input);

        input = matrix_add_bias(neuron_activation);
    }

    // calculate the derivative of the sigmoid function of the input of the result layer
    Matrix* ones = matrix_create(output[network->hidden_amount]->rows, output[network->hidden_amount]->columns);
    matrix_fill(ones, 1);
    Matrix* ones_minus_out = subtract(ones, output[network->hidden_amount]);
    Matrix* sigmoid_derivative = multiply(output[network->hidden_amount], ones_minus_out);

    // create wanted out-put matrix
    Matrix* wanted_output = matrix_create(output[network->hidden_amount]->rows, output[network->hidden_amount]->columns);
    matrix_fill(wanted_output, 0);
    wanted_output->numbers[label][0] = 1;

    // calculate difference to actual out-put
    Matrix* error = subtract(wanted_output, output[network->hidden_amount]);

    // calculate delta for output layer nodes
    Matrix* delta = multiply(sigmoid_derivative, error);

    //calculate the delta for all weights
    Matrix* previous_out_with_one = matrix_add_bias(output[network->hidden_amount - 1]);
    Matrix* transposed_previous_out_with_bias = transpose(previous_out_with_one);
    Matrix* weights_delta_matrix = dot(delta, transposed_previous_out_with_bias);


    // De-allocate stuff
    matrix_free(image_data);
    matrix_free(input);

    for (int i = 0; i < network->hidden_amount + 1; ++i) {
        matrix_free(output[i]);
    }

    matrix_free(ones);
    matrix_free(sigmoid_derivative);
    matrix_free(sigmoid_derivative);
    matrix_free(wanted_output);
    matrix_free(error);
    matrix_free(delta);
    matrix_free(previous_out_with_one);
    matrix_free(weights_delta_matrix);



}

Matrix * backPropagation(double learning_rate, Matrix* weights, Matrix* biases, Matrix* current_layer_activation, Matrix* previous_layer_activation, Matrix* sigma_old) {

    return NULL;
}

double sigmoid(double input) {
    return 1.0 / (1 + exp(-1 * input));
}

double square(double input) {
    return input * input;
}