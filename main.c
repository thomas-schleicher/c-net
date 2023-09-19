#include <stdio.h>
#include <stdio.h>
#include <malloc.h>

#include "matrix.h"
#include "image.h"

int main() {
    Image** images = import_images("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", NULL, 2);
    img_visualize(images[1]);

}