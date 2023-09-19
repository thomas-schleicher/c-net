#pragma once

typedef struct {
    Matrix* pixel_values;
    int image_label;
} Image;

Img** import_images(char* image_file_string, char* label_file_string, int number_of_images);
void img_print (Img* img);
void img_free (Img* img);