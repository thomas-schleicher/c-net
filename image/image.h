#include "../matrix/matrix.h"

typedef struct {
    Matrix* pixel_values;
    char label;
} Image;

typedef struct {
    const Image* image;
    const size_t size;
} Image_Container;

static const int MAGIC_NUMBER_LABEL = 2049;
static const int MAGIC_NUMBER_IMAGES = 2051;

Image ** import_images(char* image_file_string, char* label_file_string, int* number_imported, int count);
Image * load_pgm_image(char * image_file_string);
void img_print (Image* image);
void img_visualize(Image*image);
void img_free (Image* image);
void images_free (Image** images, int quantity);