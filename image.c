#pragma once

#include <stdio.h>

#include "image.h"
#include "matrix.h"

typedef struct {
    Matrix* pixel_values;
    int image_label;
} Image;

void img_print (Img* img){
    //print the image
    matrix_print(img->pixel_values);
    //print the number of the image
    printf("%d", img->image_label);
}
void img_free (Img* img)
