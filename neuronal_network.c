#include <stdlib.h>
#include "neuronal_network.h"
#include <stdio.h>
#include <time.h>

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


void save_network(Neural_Network* network) {

    // create file name and file string
    time_t seconds;
    time(&seconds);
    char* file_name = "../networks/";
    sprintf(file_name, "%ld", seconds);

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
    matrix_save(network->weights_output, file_name);

    printf("Network Saved!");
}

Neural_Network* load_network(char* file) {
    return NULL;
}

//double predict_images(Neural_Network* network, Image** images, int amount) {
//    int num_correct = 0;
//    for (int i = 0; i < amount; i++) {
//        Matrix* prediction = predict_image(network, images[i]);
//        if (matrix_argmax(prediction) == images[i]->label) {
//            num_correct++;
//        }
//        matrix_free(prediction);
//    }
//    return 1.0 * num_correct / amount;
//}

//Matrix* predict_image(Neural_Network* network, Image*);

//Matrix* predict(Neural_Network* network, Matrix* image_data) {
//    Matrix* hidden1_outputs = apply(relu, add(dot(network->weights_1, image_data), network->bias_1));
//
//    Matrix* hidden2_outputs = apply(relu, add(dot(network->weights_2, hidden1_outputs), network->bias_2));
//
//    Matrix* hidden3_outputs = apply(relu, add(dot(network->weights_3, hidden2_outputs), network->bias_3));
//
//    Matrix* final_outputs = apply(relu, dot(network->weights_output, hidden3_outputs));
//
//    Matrix* result = softmax(final_outputs);
//
//    matrix_free(hidden1_outputs);
//    matrix_free(hidden2_outputs);
//    matrix_free(hidden3_outputs);
//    matrix_free(final_outputs);
//
//    return result;
//}

void train_network(Neural_Network* network, Matrix* input, Matrix* output);
void batch_train_network(Neural_Network* network, Image** images, int size);