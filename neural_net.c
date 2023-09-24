//
// Created by jakob on 22.09.2023.
//
#include <stdarg.h>
#include <stdlib.h>
#include "neural_net.h"
#include <math.h>
#include "image.h"

//this is a helper struct only used for training.
typedef struct{
    int layer_count;
    Matrix ** weights_delta;
    Matrix ** biases_delta;
    Matrix ** sum_weights_delta;
    Matrix ** sum_biases_delta;
    Matrix ** layer_activations;
    Matrix ** layer_activations_wo_sigmoid;
} DynamicTrainingContainer;

DynamicTrainingContainer * init_training_container(Neural_Network * network){
    DynamicTrainingContainer * container = malloc(sizeof(DynamicTrainingContainer));
    container->layer_count = network->layer_count;
    container->weights_delta = malloc(sizeof(Matrix*)*network->layer_count - 1);
    container->biases_delta = malloc(sizeof(Matrix*)*network->layer_count - 1);
    container->sum_weights_delta = malloc(sizeof(Matrix*)*network->layer_count - 1);
    container->sum_biases_delta = malloc(sizeof(Matrix*)*network->layer_count - 1);
    container->layer_activations_wo_sigmoid = malloc(sizeof(Matrix*) * network->layer_count - 1);

    container->layer_activations = malloc(sizeof(Matrix*) * network->layer_count);

    for(int i = 0; i < network->layer_count-1; i++){
        container->weights_delta[i] = matrix_create(network->weights[i]->rows, network->weights[i]->columns);
        container->biases_delta[i] = matrix_create(network->biases[i]->rows, network->biases[i]->columns);
        container->sum_weights_delta[i] = matrix_create(network->weights[i]->rows, network->weights[i]->columns);
        container->sum_biases_delta[i] = matrix_create(network->biases[i]->rows, network->biases[i]->columns);
        container->layer_activations_wo_sigmoid[i] = matrix_create(network->sizes[i], 1);
    }
    for (int i = 0; i < network->layer_count; i++) {
        container->layer_activations[i] = matrix_create(network->sizes[i], 1);
    }
    return container;
}

void dynamic_training_container_reset_delta(DynamicTrainingContainer * container){
    for(int i = 0; i < container->layer_count-1; i++){
        matrix_fill(container->weights_delta[i], 0);
        matrix_fill(container->biases_delta[i], 0);
    }
}

void dynamic_training_container_reset_sum_delta(DynamicTrainingContainer * container){
    for(int i = 0; i < container->layer_count-1; i++){
        matrix_fill(container->sum_weights_delta[i], 0);
        matrix_fill(container->sum_biases_delta[i], 0);
    }
}

void dynamic_training_container_free_everything(DynamicTrainingContainer * container){

    for(int i = 0; i < container->layer_count-1; i++){
        matrix_free(container->weights_delta[i]);
        matrix_free(container->biases_delta[i]);
        matrix_free(container->sum_weights_delta[i]);
        matrix_free(container->sum_biases_delta[i]);
        matrix_free(container->layer_activations_wo_sigmoid[i]);
    }
    for (int i = 0; i < container->layer_count; i++) {
        matrix_free(container->layer_activations[i]);
    }

    free(container->weights_delta);
    free(container->biases_delta);
    free(container->sum_weights_delta);
    free(container->sum_biases_delta);
    free(container->layer_activations_wo_sigmoid);

    free(container->layer_activations);
}


void evaluate(Neural_Network * network, Image** images, int imageCount){
    int numCorrect = 0;
    for(int i = 0; i <= imageCount; i++){
        Matrix * input = matrix_flatten(images[i]->pixel_values, 0);
        Matrix * res = feedforward(network, input);
        char result = (char)matrix_argmax(res);
        if(result == images[i]->label){
            numCorrect++;
        }
        matrix_free(input);
        matrix_free(res);
    }
    printf("%d/%d", numCorrect, imageCount);
}

double sigmoid(double input) {
    return 1.0 / (1 + exp(-input));
}

double sigmoid_prime(double input){
    return sigmoid(input)*(1- sigmoid(input));
}

void back_prop(Neural_Network * network, Image* training_sample, DynamicTrainingContainer * trainingContainer){
    dynamic_training_container_reset_delta(trainingContainer);

    Matrix * desired_result = create_one_hot_result(training_sample); //freed in line 47


    //feedforward######################################
    //input_activation
    Matrix * current_activation = matrix_flatten(training_sample->pixel_values, 0);//freed by freeing layer_activation
    trainingContainer->layer_activations[0] = current_activation;

    for(int i = 0; i < network->layer_count-1; i++){
        Matrix * dot_result = dot(network->weights[i], current_activation);//freed 3 lines below
        Matrix * addition_result = add(dot_result, network->biases[i]); //freed by freeing layer activations wo sigmoid
        matrix_free(dot_result);
        trainingContainer->layer_activations_wo_sigmoid[i] = addition_result;
        current_activation = apply(sigmoid, addition_result);
        trainingContainer->layer_activations[i+1] = current_activation; //freed by freeing layer activations
        dot_result = NULL;
    }

    //backward pass####################################
    //calculate delta for last layer;
    //bias
    Matrix * subtraction_result = subtract(trainingContainer->layer_activations[network->layer_count-1], desired_result);
    Matrix * s_prime = apply(sigmoid_prime, trainingContainer->layer_activations_wo_sigmoid[network->layer_count-2]);
    Matrix * delta = multiply(subtraction_result, s_prime);
    matrix_free(s_prime);
    matrix_free(subtraction_result);
    trainingContainer->biases_delta[network->layer_count-2] = delta;

    //weights
    Matrix * transposed = transpose(trainingContainer->layer_activations[network->layer_count-2]);
    trainingContainer->weights_delta[network->layer_count-2] = dot(delta, transposed);
    matrix_free(transposed);
    transposed = NULL;

    for(int layer = network->layer_count-3; layer >= 0; layer--){
        Matrix * activation_wo_sigmoid = trainingContainer->layer_activations_wo_sigmoid[layer];
        Matrix * derivative = apply(sigmoid_prime, activation_wo_sigmoid);
        Matrix * transposed_layer_weight = transpose(network->weights[layer + 1]);
        Matrix * dot_result = dot(transposed_layer_weight, delta);
        matrix_free(transposed_layer_weight);
        delta = multiply(dot_result, derivative);

        trainingContainer->biases_delta[layer] = delta;
        Matrix * transposed_activation = transpose(trainingContainer->layer_activations[layer]);
        trainingContainer->weights_delta[layer] = dot(delta, transposed_activation);
        matrix_free(transposed_activation);
    }

    matrix_free(desired_result);

}

void update_batch(Neural_Network * network, DynamicTrainingContainer * trainingContainer, Image** training_data, int batch_start, int batch_end, double learning_rate){
    dynamic_training_container_reset_delta(trainingContainer);
    dynamic_training_container_reset_sum_delta(trainingContainer);

    for(int i = batch_start; i <= batch_end; i++){
        back_prop(network, training_data[i], trainingContainer);
        for(int j = 0; j < network->layer_count-1; j++){
            Matrix * sum_weights_free = trainingContainer->sum_weights_delta[j];
            trainingContainer->sum_weights_delta[j] = add(trainingContainer->sum_weights_delta[j], trainingContainer->weights_delta[j]);
            matrix_free(sum_weights_free);

            Matrix * sum_biases_free = trainingContainer->sum_biases_delta[j];
            trainingContainer->sum_biases_delta[j] = add(trainingContainer->sum_biases_delta[j], trainingContainer->biases_delta[j]);
            matrix_free(sum_biases_free);
        }
    }

    //change network
    double scaling_factor = learning_rate/(batch_end-batch_start);
    for(int i = 0; i < network->layer_count-1; i++){
        //update weights
        Matrix * weight_change = scale(trainingContainer->sum_weights_delta[i], scaling_factor);
        Matrix * new_weights = subtract(network->weights[i], weight_change);
        matrix_free(network->weights[i]);
        network->weights[i] = new_weights;

        //update biases
        Matrix * bias_change = scale(trainingContainer->sum_biases_delta[i], scaling_factor);
        Matrix * new_biases = subtract(network->biases[i], bias_change);
        matrix_free(network->biases[i]);
        network->biases[i] = new_biases;
    }
}

void train_network_with_batches(Neural_Network * network, Image** training_data, int image_count, int epochs, int batch_size, double learning_rate){
    DynamicTrainingContainer * container = init_training_container(network);


    for(int i = 0; i < epochs;  i++){
        for(int j = 0; j < image_count/batch_size; j++){
            int batch_start = j*batch_size;
            int batch_end = j*batch_size + batch_size - 1;
            update_batch(network, container, training_data, batch_start, batch_end, learning_rate);
        }
        evaluate(network, training_data, 500);
    }

    dynamic_training_container_free_everything(container);
    free(container);
}


Neural_Network* create_network(int layer_count,...){
    Neural_Network * network = malloc(sizeof(Neural_Network));
    network->layer_count = layer_count;
    network->sizes = malloc(sizeof(int) * layer_count);
    network->weights = malloc(sizeof(Matrix*)*(layer_count-1));
    network->biases = malloc(sizeof(Matrix*)*(layer_count-1));

    //read sizes
    va_list layer_sizes;
    va_start(layer_sizes, layer_count);
    for(int i = 0; i < layer_count; i++){
        network->sizes[i] = va_arg(layer_sizes, int);
    }
    va_end(layer_sizes);

    //init weights
    for(int i = 0; i < layer_count-1; i++){
        network->weights[i] = matrix_create(network->sizes[i+1], network->sizes[i]);
        matrix_randomize(network->weights[i], network->sizes[i]);
    }

    //init biases
    for(int i = 0; i < layer_count-1; i++){
        network->biases[i] = matrix_create(network->sizes[i+1], 1);
        matrix_randomize(network->biases[i], network->sizes[i]);
    }

    return network;
}



//given an input "activations" it returns the matrix that the network would output
Matrix * feedforward(Neural_Network * net, Matrix * activations){

    Matrix * current_layer_activation = activations;

    //next_layer_activation = sigmoid(dot(layer_weights, layer_activations)+layer_biases);
    for(int i = 0; i < net->layer_count - 1; i++){
        Matrix * dot_result = dot(net->weights[i], current_layer_activation);
        Matrix * addition_result = add(dot_result, net->biases[i]);
        Matrix * sigmoid_result = apply(sigmoid, addition_result);

        current_layer_activation = sigmoid_result;
        matrix_free(dot_result);
        matrix_free(addition_result);
    }
    return current_layer_activation;
}




