
#include "neuronal_network.h"
#include <stdio.h>
#include <time.h>

Neural_Network* new_network(int input_size, int hidden_size, int output_size, double learning_rate);
void print_network(Neural_Network* network);
void free_network(Neural_Network* network);

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
    fclose(file_name);

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

}

double predict_images(Neural_Network* network, Image** images, int amount);
Matrix* predict_image(Neural_Network* network, Image*);
Matrix* predict(Neural_Network* network, Matrix* image_data);

void train_network(Neural_Network* network, Matrix* input, Matrix* output);
void batch_train_network(Neural_Network* network, Image** images, int size);