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
    // initialize networks variables
    network->hidden_size = hidden_size;
    network->input_size = input_size;
    network->output_size = output_size;
    network->learning_rate = learning_rate;

    Matrix** weights = malloc(sizeof(Matrix*)*(hidden_amount + 1));
    network->weights = weights;

    network->weights[0] = matrix_create(hidden_size, input_size+1);
    for(int i=1;i<hidden_amount;i++){
        network->weights[i] = matrix_create(hidden_size, hidden_size+1);
    }
    network->weights[hidden_amount] = matrix_create(output_size, hidden_size);

    return network;
}

void randomize_network(Neural_Network* network, int scope){
    matrix_randomize(network->weights_1, scope);
    matrix_randomize(network->weights_2, scope);
    matrix_randomize(network->weights_3, scope);
    matrix_randomize(network->weights_output, scope);
//    matrix_randomize(network->bias_1, scope);
//    matrix_randomize(network->bias_2, scope);
//    matrix_randomize(network->bias_3, scope);
//    matrix_randomize(network->bias_output, scope);

    matrix_fill(network->bias_1, 1);
    matrix_fill(network->bias_2, 1);
    matrix_fill(network->bias_3, 1);
    matrix_fill(network->bias_output, 1);
}

void free_network(Neural_Network* network){
    matrix_free(network->weights_1);
    matrix_free(network->weights_2);
    matrix_free(network->weights_3);
    matrix_free(network->weights_output);
    matrix_free(network->bias_1);
    matrix_free(network->bias_2);
    matrix_free(network->bias_3);
    matrix_free(network->bias_output);
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
    fprintf(save_file, "%d\n", network->output_size);

    // close the file
    fclose(save_file);

    // save first layer
    matrix_save(network->bias_1, file_name);
    matrix_save(network->weights_1, file_name);

    // save second layer
    matrix_save(network->bias_2, file_name);
    matrix_save(network->weights_2, file_name);

    // save third layer
    matrix_save(network->bias_3, file_name);
    matrix_save(network->weights_3, file_name);

    // save output weights
    matrix_save(network->bias_output, file_name);
    matrix_save(network->weights_output, file_name);

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
    int output_size = (int) strtol(buffer, NULL, 10);

    // create a new network to fill with the saved data
    Neural_Network* saved_network = new_network(input_size, hidden_size, output_size, 0);

    // load matrices from file into struct
    saved_network->bias_1 = load_next_matrix(save_file);
    saved_network->weights_1 = load_next_matrix(save_file);
    saved_network->bias_2 = load_next_matrix(save_file);
    saved_network->weights_2 = load_next_matrix(save_file);
    saved_network->bias_3 = load_next_matrix(save_file);
    saved_network->weights_3 = load_next_matrix(save_file);
    saved_network->bias_output = load_next_matrix(save_file);
    saved_network->weights_output = load_next_matrix(save_file);

    // return saved network
    fclose(save_file);
    return saved_network;
}

void print_network(Neural_Network* network) {
    matrix_print(network->bias_1);
    matrix_print(network->bias_2);
    matrix_print(network->bias_3);
    matrix_print(network->bias_output);

    matrix_print(network->weights_1);
    matrix_print(network->weights_2);
    matrix_print(network->weights_3);
    matrix_print(network->weights_output);
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
    Matrix* h1_dot = dot(network->weights_1, image_data);
    Matrix* h1_add = add(h1_dot, network->bias_1);
    Matrix* h1_outputs = apply(sigmoid, h1_add);

    Matrix* h2_dot = dot(network->weights_2, h1_outputs);
    Matrix* h2_add = add(h2_dot, network->bias_2);
    Matrix* h2_outputs = apply(sigmoid, h2_add);

    Matrix* h3_dot = dot(network->weights_3, h2_outputs);
    Matrix* h3_add = add(h3_dot, network->bias_3);
    Matrix* h3_outputs = apply(sigmoid, h3_add);

    Matrix* final_dot = dot(network->weights_output, h3_outputs);
    Matrix* final_add = add(final_dot, network->bias_output);
    Matrix* final_outputs = apply(sigmoid, final_add);

    matrix_free(h1_dot);
    matrix_free(h1_add);
    matrix_free(h1_outputs);

    matrix_free(h2_dot);
    matrix_free(h2_add);
    matrix_free(h2_outputs);

    matrix_free(h3_dot);
    matrix_free(h3_add);
    matrix_free(h3_outputs);

    matrix_free(final_dot);
    matrix_free(final_add);

    return final_outputs;
}

void train_network(Neural_Network* network, Image *image, int label) {

    Matrix* input = matrix_flatten(image->pixel_values, 0);

    // Forward Pass
    Matrix* h1_dot = dot(network->weights_1, input);
    Matrix* h1_add = add()



    matrix_free(input);
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