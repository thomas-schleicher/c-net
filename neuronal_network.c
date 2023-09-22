#include <stdlib.h>
#include "neuronal_network.h"
#include <stdio.h>
#include <time.h>
#include <math.h>

double sigmoid(double input);
double sigmoid_derivative(double x);

Matrix* softmax(Matrix* matrix);
double square(double input);

double loss_function(Matrix* output_matrix, int image_label);

void backPropagation(double learning_rate, Matrix* weights, Matrix* biases, Matrix* current_layer_activation, Matrix* previous_layer_activation, Matrix* sigma_old);

Neural_Network* new_network(int input_size, int hidden_size, int output_size, double learning_rate){
    Neural_Network *network = malloc(sizeof(Neural_Network));
    // initialize networks variables
    network->hidden_size = hidden_size;
    network->input_size = input_size;
    network->output_size = output_size;
    network->learning_rate = learning_rate;

    network->weights_1 = matrix_create(hidden_size, input_size);
    network->weights_2 = matrix_create(hidden_size, hidden_size);
    network->weights_3 = matrix_create(hidden_size, hidden_size);
    network->weights_output = matrix_create(output_size, hidden_size);
    network->bias_1 = matrix_create(hidden_size, 1);
    network->bias_2 = matrix_create(hidden_size, 1);
    network->bias_3 = matrix_create(hidden_size, 1);
    network->bias_output = matrix_create(output_size, 1);

    return network;
}

void randomize_network(Neural_Network* network, int scope){
    matrix_randomize(network->weights_1, scope);
    matrix_randomize(network->weights_2, scope);
    matrix_randomize(network->weights_3, scope);
    matrix_randomize(network->weights_output, scope);
    matrix_randomize(network->bias_1, scope);
    matrix_randomize(network->bias_2, scope);
    matrix_randomize(network->bias_3, scope);
    matrix_randomize(network->bias_output, scope);
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

double measure_network_accuracy(Neural_Network* network, Image** images, int amount) {
    int num_correct = 0;
    for (int i = 0; i < amount; i++) {
        Matrix* prediction = predict_image(network, images[i]);
        if (matrix_argmax(prediction) == images[i]->label) {
            num_correct++;
        }
        matrix_free(prediction);
    }
    return 1.0 * num_correct / amount;
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

    Matrix* result = softmax(final_outputs);

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
    matrix_free(final_outputs);

    return result;
}

double cost_function(Matrix* calculated, int expected){
    calculated->numbers[expected] -= 1;
    apply(square, calculated);

//    double loss = 0.5 * (target - output) * (target - output);

    return 0;
}

void train_network(Neural_Network* network, Image *image, int label) {

    // Flatten the image into matrix
    Matrix* input = matrix_flatten(image->pixel_values, 0);

    // Perform forward propagation
    Matrix* h1_dot = dot(network->weights_1, input);
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

    // begin backpropagation
    Matrix* sigma = matrix_create(final_outputs->rows, 1);
    matrix_fill(sigma, 1);
    Matrix* temp1 = subtract(sigma, final_outputs);
    Matrix* temp2 = multiply(temp1, final_outputs); // * soll-ist
    Matrix* temp3 = matrix_create(final_outputs->rows, final_outputs->columns);
    matrix_fill(temp3, 0);
    temp3->numbers[label][0] = 1;
    Matrix* temp4 = subtract(temp3, final_outputs);
    sigma = multiply(temp2, temp4);

    Matrix* temp5 = transpose(h3_outputs);
    Matrix* temp6 = dot(sigma, temp5);
    Matrix* weights_delta = scale(temp6, network->learning_rate);
    Matrix* bias_delta = scale(sigma, network->learning_rate);

    Matrix* temp7 = add(weights_delta, network->weights_output);
    matrix_free(network->weights_output);
    network->weights_output = temp7;

    Matrix* temp8 = add(bias_delta, network->bias_output);
    matrix_free(network->bias_output);
    network->bias_output = temp8;

    // other levels
    backPropagation(network->learning_rate, network->weights_3, network->bias_3, h3_outputs, h2_outputs, sigma);
    backPropagation(network->learning_rate, network->weights_2, network->bias_2, h2_outputs, h1_outputs, sigma);
    backPropagation(network->learning_rate, network->weights_1, network->bias_1, h1_outputs, input, sigma);

    matrix_free(input);

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
    matrix_free(final_outputs);

    matrix_free(weights_delta);
    matrix_free(bias_delta);

    matrix_free(sigma);

    matrix_free(temp1);
    matrix_free(temp2);
    matrix_free(temp3);
    matrix_free(temp4);
    matrix_free(temp5);
    matrix_free(temp6);
    matrix_free(temp7);
    matrix_free(temp8);
}

void backPropagation(double learning_rate, Matrix* weights, Matrix* biases, Matrix* current_layer_activation, Matrix* previous_layer_activation, Matrix* sigma_old) {
    Matrix* sigma_new = matrix_create(current_layer_activation->rows, 1);
    matrix_fill(sigma_new, 1);

    Matrix* temp1 = subtract(sigma_new, current_layer_activation);
    Matrix* temp2 = multiply(temp1, current_layer_activation); // *sum(delta*weights)

    for(int i = 0; i < current_layer_activation->rows; i++) {
        double sum = 0;
        for (int j = 0; j < sigma_old->rows; j++) {
            sum += current_layer_activation->numbers[i][j] * sigma_old->numbers[j][0];
        }
        temp1->numbers[i][0] = sum;
    }
    sigma_new = multiply(temp2, temp1);

    // new sigma done

    Matrix* temp3 = transpose(previous_layer_activation);
    Matrix* temp4 = dot(sigma_new, temp3);
    Matrix* weights_delta = scale(temp4, learning_rate);
    Matrix* bias_delta = scale(sigma_new, learning_rate);

    Matrix* temp5 = add(weights_delta, weights);
    free(weights->numbers);
    weights->numbers = temp5->numbers;

    Matrix* temp6 = add(bias_delta, biases);
    free(biases->numbers);
    biases->numbers = temp6->numbers;

    sigma_old->rows = sigma_new->rows;
    sigma_old->columns = sigma_new->columns;
    free(sigma_old->numbers);
    sigma_old->numbers = sigma_new->numbers;

    free(sigma_new);

    matrix_free(temp1);
    matrix_free(temp2);
    matrix_free(temp3);
    matrix_free(temp4);
    matrix_free(temp5);
    matrix_free(temp6);
    matrix_free(weights_delta);
    matrix_free(bias_delta);
}


//void batch_train_network(Neural_Network* network, Image** images, int size);

double sigmoid(double input) {
    return 1.0 / (1 + exp(-1 * input));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

Matrix* softmax(Matrix* matrix) {
    double total = 0;

    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            total += exp(matrix->numbers[i][j]);
        }
    }
    Matrix* result_matrix = matrix_create(matrix->rows, matrix->columns);
    for (int i = 0; i < result_matrix->rows; i++) {
        for (int j = 0; j < result_matrix->columns; j++) {
            result_matrix->numbers[i][j] = exp(matrix->numbers[i][j]) / total;
        }
    }
    return result_matrix;
}

double square(double input) {
    return input * input;
}

double loss_function(Matrix* output_matrix, int image_label) {
    Matrix* temp = matrix_copy(output_matrix);

    temp->numbers[1, image_label] -= 1;
    apply(square, temp);

    matrix_free(temp);

    return matrix_sum(temp);;
}