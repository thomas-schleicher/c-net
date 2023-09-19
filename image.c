#pragma once

#include <stdlib.h>
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
    //print the number of the image
    printf("Number it is supposed to be: %d\n", img->image_label);
}

void img_free (Img* img) {
    //frees the matrix of image (deep free)
    matrix_free(img->pixel_values);
    //frees the rest of img
    free(img);
}

void images_free (Img** images, int quantity){
    //frees every single image
    for(int i=0;i<quantity;i++){
        img_free(images[i]);
    }
    //frees the rest of images
    free(images);
}
