#pragma once
#include "matrix.h"

typedef struct {
    Matrix* pixel_values;
    char label;
} Image;


static const int MAGIC_NUMBER_LABEL = 2049;

static const int MAGIC_NUMBER_IMAGES = 2051;

/**
 * reads a specified number of images out of the training dataset
 * @param image_file_string Path to the file containing the image data
 * @param label_file_string Path to the file containing the image labels
 * @param ptr via this pointer, the images can be accessed
 * @param count maximum number of images to be loaded. If it is 0, all available images are loaded.
 * @return
 */
Image ** import_images(char* image_file_string, char* label_file_string, int* number_imported, int count);
Image * load_pgm_image(char * image_file_string);
Matrix* create_one_hot_result(Image* image);
void img_print (Image* image);
void img_visualize(Image*image);
void img_free (Image* image);