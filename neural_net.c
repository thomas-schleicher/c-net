//
// Created by jakob on 22.09.2023.
//
#include <stdarg.h>
#include <stdlib.h>
#include "neural_net.h"
#include <math.h>
#include "image.h"

typedef struct{
    Neural_Network * network;
    Matrix ** weights_delta;
    Matrix ** biases_delta
};


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

void back_prop(Neural_Network * network, Image* training_sample, Matrix ** weights_delta, Matrix ** biases_delta){
    //all Matrix** are external, to avoid repeated memory allocation and deallocation.
    for(int i = 0; i < network->layer_count - 1; i++){
        matrix_fill(weights_delta[i], 0);
        matrix_fill(biases_delta[i], 0);
    }

    Matrix * desired_result = create_one_hot_result(training_sample); //freed in line 47


    //feedforward######################################
    //input_activation
    Matrix * current_activation = matrix_flatten(training_sample->pixel_values, 0);//freed by freeing layer_activation

    Matrix ** layer_activations = malloc(sizeof(Matrix*) * network->layer_count); //freed at end
    Matrix ** layer_activations_wo_sigmoid = malloc(sizeof(Matrix*) * network->layer_count - 1);//freed at end
    layer_activations[0] = current_activation;

    for(int i = 0; i < network->layer_count-1; i++){
        Matrix * dot_result = dot(network->weights[i], current_activation);//freed 3 lines below
        Matrix * addition_result = add(dot_result, network->biases[i]); //freed by freeing layer activations wo sigmoid
        matrix_free(dot_result);
        layer_activations_wo_sigmoid[i] = addition_result;
        current_activation = apply(sigmoid, addition_result);
        layer_activations[i+1] = current_activation; //freed by freeing layer activations
        dot_result = NULL;
    }

    //backward pass####################################
    //calculate delta for last layer;
    //bias
    Matrix * subtraction_result = subtract(layer_activations[network->layer_count-1], desired_result);
    Matrix * s_prime = apply(sigmoid_prime, layer_activations_wo_sigmoid[network->layer_count-2]);
    Matrix * delta = multiply(subtraction_result, s_prime);
    matrix_free(s_prime);
    matrix_free(subtraction_result);
    biases_delta[network->layer_count-2] = delta;

    //weights
    Matrix * transposed = transpose(layer_activations[network->layer_count-2]);
    weights_delta[network->layer_count-2] = dot(delta, transposed);
    matrix_free(transposed);
    transposed = NULL;

    for(int layer = network->layer_count-3; layer >= 0; layer--){
        Matrix * activation_wo_sigmoid = layer_activations_wo_sigmoid[layer];
        Matrix * derivative = apply(sigmoid_prime, activation_wo_sigmoid);
        Matrix * transposed_layer_weight = transpose(network->weights[layer + 1]);
        Matrix * dot_result = dot(transposed_layer_weight, delta);
        matrix_free(transposed_layer_weight);
        delta = multiply(dot_result, derivative);

        biases_delta[layer] = delta;
        Matrix * transposed_activation = transpose(layer_activations[layer]);
        weights_delta[layer] = dot(delta, transposed_activation);
        matrix_free(transposed_activation);
    }

    matrix_free(desired_result);

    //free layer_activations
    for(int i = 0; i < network->layer_count; i++){
        matrix_free(layer_activations[i]);
    }
    free(layer_activations);

    //free layer_activations wo sigmoid
    for(int i = 0; i < network->layer_count - 1; i++){
        matrix_free(layer_activations_wo_sigmoid[i]);
    }
    free(layer_activations_wo_sigmoid);


}

void update_batch(Neural_Network * network, Image** training_data, int batch_start, int batch_end, double learning_rate){
    Matrix** weights_delta = malloc(sizeof(Matrix*)*network->layer_count - 1);
    Matrix** biases_delta = malloc(sizeof(Matrix*)*network->layer_count - 1);
    Matrix** sum_weights_delta = malloc(sizeof(Matrix*)*network->layer_count - 1);
    Matrix** sum_biases_delta = malloc(sizeof(Matrix*)*network->layer_count - 1);

    for(int i = 0; i < network->layer_count - 1; i++){
        weights_delta[i] = matrix_create(network->weights[i]->rows, network->weights[i]->columns);
        biases_delta[i] = matrix_create(network->biases[i]->rows, network->biases[i]->columns);
        sum_weights_delta[i] = matrix_create(network->weights[i]->rows, network->weights[i]->columns);
        sum_biases_delta[i] = matrix_create(network->biases[i]->rows, network->biases[i]->columns);
    }

    for(int i = batch_start; i <= batch_end; i++){
        back_prop(network, training_data[i], weights_delta, biases_delta);
        for(int j = 0; j < network->layer_count-1; j++){
            Matrix * sum_weights_free = sum_weights_delta[j];
            sum_weights_delta[j] = add(sum_weights_delta[j], weights_delta[j]);
            matrix_free(sum_weights_free);

            Matrix * sum_biases_free = sum_biases_delta[j];
            sum_biases_delta[j] = add(sum_biases_delta[j], biases_delta[j]);
            matrix_free(sum_biases_free);
        }
    }

    //change network
    double scaling_factor = learning_rate/(batch_end-batch_start);
    for(int i = 0; i < network->layer_count-1; i++){
        //update weights
        Matrix * weight_change = scale(sum_weights_delta[i], scaling_factor);
        matrix_free(sum_weights_delta[i]);
        Matrix * new_weights = subtract(network->weights[i], weight_change);
        matrix_free(network->weights[i]);
        network->weights[i] = new_weights;

        //update biases
        Matrix * bias_change = scale(sum_biases_delta[i], scaling_factor);
        matrix_free(sum_biases_delta[i]);
        Matrix * new_biases = subtract(network->biases[i], bias_change);
        matrix_free(network->biases[i]);
        network->biases[i] = new_biases;
    }
    free(sum_weights_delta);
    free(sum_biases_delta);
    for(int i = 0; i < network->layer_count - 1; i++){
        matrix_free(weights_delta[i]);
        matrix_free(biases_delta[i]);
    }


}

void train_network_with_batches(Neural_Network * network, Image** training_data, int image_count, int epochs, int batch_size, double learning_rate){
    for(int i = 0; i < epochs;  i++){
        for(int j = 0; j < image_count/batch_size; j++){
            int batch_start = j*batch_size;
            int batch_end = j*batch_size + batch_size - 1;
            update_batch(network, training_data, batch_start, batch_end, learning_rate);
        }
        evaluate(network, training_data, 1000);
    }
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




