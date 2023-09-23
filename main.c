#include <stdio.h>

#include "matrix.h"
#include "image.h"

#include "neural_net.h"

int main() {
//    Image** images = import_images("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", NULL, 60000);
////    img_visualize(images[4]);
//
//    Neural_Network* nn = new_network(28*28, 16, 10, 0.5);
//    randomize_network(nn, 20);
////    save_network(nn);
//
////    Neural_Network* nn = load_network("../networks/test1.txt");
//
//
//    for (int i = 0; i < 10000; ++i) {
//        train_network(nn, images[i], images[i]->label);
//    }
//
//    measure_network_accuracy(nn, images, 100);
//    Matrix *m = matrix_create(2, 1);
//    m->numbers[0][0] = 1;
//    m->numbers[1][0] = 1;
//    Neural_Network * net = create_network(3, 2, 3, 1);
//    feedforward(net, m);
//
//    int pause;
    int imported_count = 0;
    Image** images = import_images("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", &imported_count, 10000);
    Neural_Network * net = create_network(3, 28*28, 30, 10);
    train_network_with_batches(net, images, imported_count, 1, 10, 3);
}