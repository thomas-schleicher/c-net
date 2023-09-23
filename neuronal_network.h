
#include "matrix.h"
#include "image.h"

typedef struct {
    int input_size;
    int hidden_size;
    int hidden_amount;
    int output_size;

    Matrix** weights;

    double learning_rate;

} Neural_Network;

static const int MAX_BYTES = 100;

Neural_Network* new_network(int input_size, int hidden_size, int hidden_amount, int output_size, double learning_rate);

void randomize_network(Neural_Network* network, int scope);
void free_network(Neural_Network* network);

void save_network(Neural_Network* network);
Neural_Network* load_network(char* file);

void print_network(Neural_Network* network);

double measure_network_accuracy(Neural_Network* network, Image** images, int amount);
Matrix* predict_image(Neural_Network* network, Image* image);

void train_network(Neural_Network* network, Image *image, int label);
