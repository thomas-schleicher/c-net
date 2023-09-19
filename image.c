#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "matrix.h"

Image** import_images(char* image_file_string, char* label_file_string, int number_of_images) {

    // create file pointer for the image and label data
    FILE* image_file = fopen(image_file_string, 'r');
    FILE* label_file = fopen(label_file_string, 'r');

    // check if the file could be opened
    if(image_file == NULL || label_file == NULL) {
        printf("ERROR: File could not be opened! (import_images)");
    }
    char ch;
    do {
        ch = fgetc(label_file);
        printf("%c", ch);

        // Checking if character is not EOF.
        // If it is EOF stop reading.
    } while (ch != EOF);


    // allocate memory for the storage of images
//    Image** images = malloc(sizeof(Image) * number_of_images);

    fclose(image_file);
    fclose(label_file);
}

void img_print (Image* img) {

    //print the image
    matrix_print(img->pixel_values);

    //print the label of the image
    printf("%d", img->image_label);
}

void img_free (Image* img) {

}
