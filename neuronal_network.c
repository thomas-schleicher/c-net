
#include "neuronal_network.h"

Neural_Network* new_network(int input_size, int hidden_size, int output_size, double learning_rate);
void print_network(Neural_Network* network);
void free_network(Neural_Network* network);

void save_network(Neural_Network* network, char* file);
Neural_Network* load_network(char* file);

double predict_images(Neural_Network* network, Image** images, int amount);
Matrix* predict_image(Neural_Network* network, Image*);
Matrix* predict(Neural_Network* network, Matrix* image_data);

void train_network(Neural_Network* network, Matrix* input, Matrix* output);
void batch_train_network(Neural_Network* network, Image** images, int size);