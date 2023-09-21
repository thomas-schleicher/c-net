#include <stdlib.h>
#include "neuronal_network.h"
#include <stdio.h>
#include <time.h>
#include <math.h>

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
    network.bias_output = matrix_create(output_size, 1);



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



    return saved_network;
}

double predict_images(Neural_Network* network, Image** images, int amount) {
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
    Matrix* hidden1_outputs = apply(relu, add(dot(network->weights_1, image_data), network->bias_1));

    Matrix* hidden2_outputs = apply(relu, add(dot(network->weights_2, hidden1_outputs), network->bias_2));

    Matrix* hidden3_outputs = apply(relu, add(dot(network->weights_3, hidden2_outputs), network->bias_3));

    Matrix* final_outputs = apply(relu, dot(network->weights_output, hidden3_outputs));

    Matrix* result = softmax(final_outputs);

    matrix_free(hidden1_outputs);
    matrix_free(hidden2_outputs);
    matrix_free(hidden3_outputs);
    matrix_free(final_outputs);

    return result;
}

//void train_network(Neural_Network* network, Matrix* input, Matrix* output);
//void batch_train_network(Neural_Network* network, Image** images, int size);

double relu(double input) {
    if (input < 0){
        return 0.0;
    }
    return input;
    //TODO: relu formel
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