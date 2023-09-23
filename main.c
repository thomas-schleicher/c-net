#include <stdio.h>

#include "image.h"
#include "neuronal_network.h"

int main() {
    Image** images = import_images("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", NULL, 60000);
//    img_visualize(images[0]);
//    img_visualize(images[1]);

//    matrix_print(images[0]->pixel_values);
//    matrix_print(images[1]->pixel_values);

    Neural_Network* nn = new_network(28*28, 40, 5, 10, 0.08);
    randomize_network(nn, 1);
//        Neural_Network* nn = load_network("../networks/newest_network.txt");
//    printf("Done loading!\n");

//    batch_train(nn, images, 20000, 20);

    for (int i = 0; i < 30000; ++i) {
        train_network(nn, images[i], images[i]->label);
    }

    save_network(nn);

    printf("%lf\n", measure_network_accuracy(nn, images, 10000));

}