//
// Created by jakob on 22.09.2023.
//

#include "matrix.h"
#include "image.h"

#ifndef C_NET_NEURAL_NET_H
#define C_NET_NEURAL_NET_H

#endif //C_NET_NEURAL_NET_H

typedef struct {
    int layer_count;
    int* sizes;
    Matrix ** weights;
    Matrix ** biases;
} Neural_Network;

Neural_Network* create_network(int layer_count,...);
Matrix * feedforward(Neural_Network * net, Matrix * activations);
void train_network_with_batches(Neural_Network * network, Image** training_data, int image_count, int epochs, int batch_size, double learning_rate);
