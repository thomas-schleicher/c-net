#pragma once
#include "matrix.h"

typedef struct {
    Matrix* pixel_values;
    int image_label;
} Image;

Image** import_images(char* image_file_string, char* label_file_string, int number_of_images);
void img_print (Image* image);
void img_free (Image* image);