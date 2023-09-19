#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "matrix.h"

int endian_swap(int input) {
    return ((input >> 24) & 0xff) |      // move byte 3 to byte 0
           ((input << 8) & 0xff0000) |   // move byte 1 to byte 2
           ((input >> 8) & 0xff00) |     // move byte 2 to byte 1
           ((input << 24) & 0xff000000); // byte 0 to byte 3
}

int validate_files(FILE* image_file, FILE* label_file) {

    // read magic number from files
    int magic_number_label, magic_number_images;
    fread(&magic_number_label, 4, 1, label_file);
    fread(&magic_number_images, 4, 1, image_file);

    // compare magic numbers with pre-defined value
    if(endian_swap(magic_number_label) != 2049 || endian_swap(magic_number_images) != 2051) {
        return 0;
    }

    return 1;
}

Image** import_images(char* image_file_string, char* label_file_string, int number_of_images) {

    // create file pointer for the image and label data
    FILE* image_file = fopen(image_file_string, "r");
    FILE* label_file = fopen(label_file_string, "r");

    // check if the file could be opened
    if(image_file == NULL || label_file == NULL) {
        printf("ERROR: File could not be opened! (import_images)");
        exit(1);
    }

    // check magic number of the files
    if(validate_files(image_file, label_file)) {
        printf("ERROR: File validation failed! (validate_files)");
        exit(1);
    }


    // Jakob Section


    // allocate memory for the storage of images
    Image** images = malloc(sizeof(Image) * number_of_images);

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
