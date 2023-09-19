#pragma once

#include "image.h"
#include "matrix.h"

typedef struct {
    Matrix* pixel_values;
    int image_label;
} Image;


