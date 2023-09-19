#pragma once

#include <stdio.h>
#include "image.h"
#include "matrix.h"

Img** import_images(char* image_file_string, char* label_file_string, int number_of_images) {

    // create file pointer for the image and label data
    FILE* image_file = fopen(image_file_string, 'r');
    FILE* label_file = fopen(label_file_string, 'r');

    // check if the file could be opened
    if(image_file == NULL || label_file == NULL) {
        printf("ERROR: File could not be opened! ()");
    }

}

void img_print (Img* img) {

    //print the image
    matrix_print(img->pixel_values);

    //print the label of the image
    printf("%d", img->image_label);
}

void img_free (Img* img) {

}
