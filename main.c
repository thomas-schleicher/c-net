#include <stdio.h>

#include "image.h"
#include "neuronal_network.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "util.h"

void parsingErrorPrintHelp(){
    printf("Syntax: c_net [train | predict]\n");
    printf("commands:\n");
    printf("train\t train the network\n");
    printf("predict\t load a pgm image and predict_demo the number\n");
    exit(1);
}

void parsingErrorTrain(){
    printf("invalid syntax\n");
    printf("Syntax: c_net train [path_to_train-images.idx3-ubyte] [path_to_train-labels.idx1-ubyte] [hidden_layer_count] [neurons_per_layer] [epochs] [learning_rate] [path_to_save_network]\n");
    exit(1);
}

void parsingErrorDetect(){
    printf("invalid syntax\n");
    printf("Syntax: c_net predict_demo [path_to_network] [image_file]");
}

void predict_demo(int argc, char** arguments){
    if(argc != 2) parsingErrorDetect();
    char * network_file = arguments[0];
    char * image_file = arguments[1];

    Neural_Network * nn = load_network(network_file);
    Image * image = load_pgm_image(image_file);
    Matrix * result = predict_image(nn, image);
    int predicted = matrix_argmax(result);
    printf("prediction result %d\n", predicted);
    matrix_print(result);
    matrix_free(result);
}

void train(int argc, char** arguments) {
    if (argc != 7) parsingErrorTrain();
    char *image_file = arguments[0];
    char *label_file = arguments[1];
    int hidden_count = (int) strtol(arguments[2], NULL, 10);
    int neurons_per_layer = (int) strtol(arguments[3], NULL, 10);
    int epochs = (int) strtol(arguments[4], NULL, 10);
    if (errno != 0) {
        printf("hidden_count, neurons_per_layer or epochs could not be parsed!\n");
        exit(1);
    }
    double learning_rate = strtod(arguments[5], NULL);
    if (errno != 0) {
        printf("learning_rate could not be parsed!\n");
        exit(1);
    }
    char *save_path = arguments[6];
    int imported = 0;
    Image ** images = import_images(image_file, label_file, &imported, 60000);
    Image ** evaluation_images = images+50000;

    int training_image_count = 50000;
    int testing_image_count = 10000;

    Neural_Network *nn = new_network(28 * 28, neurons_per_layer, hidden_count, 10, learning_rate);
    randomize_network(nn, 1);
    printf("training_network\n");
    for(int epoch = 1; epoch <= epochs; epoch++){
        printf("epoch %d\n", epoch);
        for (int i = 0; i < training_image_count; i++) {
            if (i % 1000 == 0) {
                updateBar(i * 100 / imported);
            }
            train_network(nn, images[i], images[i]->label);
        }
        updateBar(100);
        printf("\n");
        printf("accuracy %lf\n", measure_network_accuracy(nn, evaluation_images, testing_image_count));
    }
    printf("done training!\n");
    save_network(nn, save_path);
}

int main(int argc, char** argv) {
//    Image** images = import_images("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", NULL, 60000);
////    img_visualize(images[0]);
////    img_visualize(images[1]);
//
////    matrix_print(images[0]->pixel_values);
////    matrix_print(images[1]->pixel_values);
//
//    Neural_Network* nn = new_network(28*28, 40, 5, 10, 0.08);
//    randomize_network(nn, 1);
////        Neural_Network* nn = load_network("../networks/newest_network.txt");
////    printf("Done loading!\n");
//
////    batch_train(nn, images, 20000, 20);
//
//    for (int i = 0; i < 30000; ++i) {
//        train_network(nn, images[i], images[i]->label);
//    }
//
//    save_network(nn);
//
//    printf("%lf\n", measure_network_accuracy(nn, images, 10000));
    if(argc < 2){
        parsingErrorPrintHelp();
        exit(1);
    }
    if(strcmp(argv[1], "train") == 0){
        train(argc-2, argv+2);
        return 0;
    }
    if(strcmp(argv[1], "predict") == 0){
        predict_demo(argc - 2, argv + 2);
        return 0;
    }
    parsingErrorPrintHelp();

}