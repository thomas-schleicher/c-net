#include <stdio.h>

#include "image.h"
#include "neuronal_network.h"

int main() {
    Image** images = import_images("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", NULL, 60000);
//    img_visualize(images[0]);

    Neural_Network* nn = new_network(28*28, 50, 10, 0.01);
    randomize_network(nn, 1);
//    save_network(nn);

//    Neural_Network* nn = load_network("../networks/test1.txt");


    for (int i = 0; i < 20000; ++i) {
        train_network(nn, images[i], images[i]->label);
//        matrix_print(nn->weights_output);
    }

//    train_network(nn, images[0], images[0]->label);
//    train_network(nn, images[0], images[0]->label);

    printf("%lf\n", measure_network_accuracy(nn, images, 2000));

}