#include <stdio.h>

#include "image/image.h"
#include "neuronal_network.h"

int main() {

    const int amount_of_images_to_load = 60000;
    const int amount_of_images_used_to_train = 30000;
    const int amount_of_images_used_to_test = 1000;
    const int input_size = 28*28;
    const int hidden_layer_size = 50;
    const int hidden_layer_count = 3;
    const double learning_rate = 0.1;

    /*
     * Loading Images from Dataset
     */

    Image** images = import_images("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", NULL, amount_of_images_to_load);

//    img_visualize(images[0]);
//    img_print(images[0]);

    /*
     * Create a new network and randomize the weights
     */

    Neural_Network* network = new_network(input_size, hidden_layer_size, hidden_layer_count, 10, learning_rate);
    randomize_network(network, 1);

    /*
     * Training
     */

    for (int i = 0; i < amount_of_images_used_to_train; i++) {
        train_network(network, images[i], images[i]->label);
    }

    // Batch training works if you change the train_network method, but the results are not that good (needs further testing)
    // batch_train(nn, images, 30000, 2);

    printf("Trinaing Done!\n");

    /*
     * Saving and Loading
     */

//    save_network(network);
//    Neural_Network* network = load_network("../networks/newest_network.txt");

    /*
     * Measure Accuracy & predict single images
     */

    printf("Accuracy: %lf\n", measure_network_accuracy(network, images, amount_of_images_used_to_test));

//    matrix_print(predict_image(network, images[0]));

    images_free(images, amount_of_images_to_load);
    free_network(network);

    return 0;
}
