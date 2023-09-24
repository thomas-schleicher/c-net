#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "matrix.h"
#include "util.h"

void big_endian_to_c_uint(const char * bytes, void * target, int size) {
    char* helper = (char*)target;
    for(int i = 0; i < size; i++){
        *(helper+i) = *(bytes+size-i-1);
    }
}

void read_until_space_or_newline(char * buff, int maxCount, FILE * fptr){
    int bufferOffset = 0;
    char c;
    int counter = 0;
    do{
        c = (char)getc(fptr);
        buff[bufferOffset++] = c;

    }while(!feof(fptr) && c != 0 && c != ' ' && c !='\n' && counter++ < maxCount);
    buff[bufferOffset-1] = 0;
}

Image * load_pgm_image(char * image_file_string){
    FILE * fptr = fopen(image_file_string, "r");
    Image *image = malloc(sizeof(Image));
    image->label = -1;


    char buffer[2048];
    fgets(buffer, 4, fptr);
    if(buffer[0] != 'P' || buffer[1] != '5'){
        printf("Wrong file Format");
        exit(1);
    }
    if(fgetc(fptr) ==  '#'){
        fgets(buffer, 1024, fptr);
    }

    int image_width, image_height, image_white ;
    read_until_space_or_newline(buffer, 10, fptr);
    image_width = (int)strtol(buffer, NULL, 10);

    read_until_space_or_newline(buffer, 10, fptr);
    image_height = (int)strtol(buffer, NULL, 10);

    read_until_space_or_newline(buffer, 10, fptr);
    image_white = (int)strtol(buffer, NULL, 10);


    image->pixel_values = matrix_create(image_height, image_width);
    for(int i = 0; i < image_height; i++){
        fread(buffer, 1, 28, fptr);
        for(int j = 0; j < image_width;  j++){
            image->pixel_values->numbers[i][j] = (image_white - (unsigned char)buffer[j]) / 255.0;
        }
    }

    fclose(fptr);
    return image;
}


Image** import_images(char* image_file_string, char* label_file_string, int* _number_imported, int count) {
    printf("Loading Images\n");
    // create file pointer for the image and label data
    FILE* image_file = fopen(image_file_string, "rb");
    FILE* label_file = fopen(label_file_string, "rb");

    // check if the file could be opened
    if(image_file == NULL || label_file == NULL) {
        printf("ERROR: File could not be opened! (import_images)");
        exit(1);
    }

    // check magic number of the files
    char word_buffer[4];
    int buffer_size = sizeof(word_buffer);

    int magic_number_label, magic_number_images, label_count, image_count;

    //Read description of label file
    fread(word_buffer, buffer_size, 1, label_file);
    big_endian_to_c_uint(word_buffer, &magic_number_label, buffer_size);

    fread(word_buffer, 4, 1, label_file);
    big_endian_to_c_uint(word_buffer, &label_count, buffer_size);

    //Read description of file
    fread(word_buffer, 4, 1, image_file);
    big_endian_to_c_uint(word_buffer, &magic_number_images, buffer_size);

    fread(word_buffer, 4, 1, image_file);
    big_endian_to_c_uint(word_buffer, &image_count, buffer_size);

    // compare magic numbers with pre-defined value
    if(magic_number_label != MAGIC_NUMBER_LABEL || magic_number_images != MAGIC_NUMBER_IMAGES) {
        printf("TrainingData or Labels are malformed. Exiting...");
        exit(1);
    }
    if(label_count != image_count){
        printf("Number of images and labels does not match. Exiting...");
        exit(1);
    }

    if(count <= 0){
        count = image_count;
    }

    if(count > image_count){
        count = image_count;
        printf("Number of images exceeds number of available images. Loading all available images");
    }

    int image_height, image_width, image_length;
    //read image dimensions;
    fread(word_buffer, 4, 1, image_file);
    big_endian_to_c_uint(word_buffer, &image_height, buffer_size);

    fread(word_buffer, 4, 1, image_file);
    big_endian_to_c_uint(word_buffer, &image_width, 4);

    image_length = image_height*image_width;

    // allocate memory for the storage of images
    Image** images = malloc(sizeof(Image*) * count);
    if(!images){
        printf("not enough memory");
        exit(1);
    }

    unsigned char byteBuffer[image_length];
    for(int i = 0; i < count; i++){
        if(i%1000 == 0){
            updateBar(i*100/count);
        }
        images[i] = malloc(sizeof(Image));
        fread(&images[i]->label, 1, 1, label_file);
        fread(&byteBuffer, image_width*image_height, 1, image_file);
        images[i]->pixel_values = matrix_create(image_height, image_width);

        for(int j = 0; j < image_length; j++) {
            images[i]->pixel_values->numbers[j / image_width][j % image_width] = byteBuffer[j] / 255.0;
        }
    }

    if(_number_imported != NULL)*_number_imported = count;

    fclose(image_file);
    fclose(label_file);

    updateBar(100);
    printf("\n");
    return images;
}

void img_print (Image* img) {

    //print the image
    matrix_print(img->pixel_values);
    //print the number of the image
    printf("Number it is supposed to be: %d\n", img->label);
}

void img_visualize(Image* img){
    for(int i = 0; i < img->pixel_values->rows; i++){
        for(int j = 0; j < img->pixel_values->columns; j++){
            img->pixel_values->numbers[i][j] > 0.5 ? putc('#', stdout) : putc(' ', stdout);
        }
        putc('\n', stdout);
    }
    printf("Should be %d\n", img->label);
}

void img_free (Image* img) {
    //frees the matrix of image (deep free)
    matrix_free(img->pixel_values);
    //frees the rest of img
    free(img);
}

void images_free (Image** images, int quantity){
    //frees every single image
    for(int i=0;i<quantity;i++){
        img_free(images[i]);
    }
    //frees the rest of images
    free(images);
}
