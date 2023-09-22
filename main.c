#include <stdio.h>

#include "matrix.h"
#include "image.h"
#include "neuronal_network.h"

int main() {
    Image** images = import_images("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", NULL, 60000);
//    img_visualize(images[4]);

    Neural_Network* nn = new_network(28*28, 100, 10, 0.5);
    randomize_network(nn, 20);
//    save_network(nn);

//    Neural_Network* nn = load_network("../networks/test1.txt");


    for (int i = 0; i < 10000; ++i) {
        train_network(nn, images[i], images[i]->label);
    }

    printf("%lf\n", measure_network_accuracy(nn, images, 100));

}