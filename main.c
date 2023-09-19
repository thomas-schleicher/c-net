#include <stdio.h>

#include "matrix.h"
#include "image.h"
#include "neuronal_network.h"

int main() {
//    Image** images = import_images("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", NULL, 2);
//    img_visualize(images[1]);

//    Neural_Network* nn = new_network(4, 2, 3, 0.5);
//
//    int n = 20;
//
//    matrix_randomize(nn->bias_1, n);
//    matrix_randomize(nn->bias_2, n);
//    matrix_randomize(nn->bias_3, n);
//
//    matrix_randomize(nn->weights_1, n);
//    matrix_randomize(nn->weights_2, n);
//    matrix_randomize(nn->weights_3, n);
//
//    matrix_randomize(nn->weights_output, n);
//
//    save_network(nn);

    Neural_Network* nn = load_network("../networks/test1.txt");

}