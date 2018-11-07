// Reference http://yann.lecun.com/exdb/mnist/ as of Oct 23, 2018
//
// This reads the MNIST labelled digit database and prints the first few samples
//
// To compile: gcc readMNIST.c -o /tmp/readMNIST
//
// To run: /tmp/readMNIST train
//     or: /tmp/readMNIST t10k
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Read all the labels
static int readLabels(char *labelName, int *numLabels, unsigned char **buffer) {
  FILE *labelFile;
  unsigned char buf[8];
  size_t read;
  
  // Open label file for reading
  labelFile = fopen(labelName, "r");
  if (labelFile == NULL) {
    fprintf(stderr, "Unable to read %s\n", labelName);
    return 0;
  }
  
  // Read header
  read = fread(buf, sizeof(unsigned char), 8, labelFile);
  if (read != 8 || buf[0] != 0 || buf[1] != 0 || buf[2] != 8 || buf[3] != 1) {
    fprintf(stderr, "Invalid header in file %s, read=%ld\n", labelName, (long)read);
    return 0;
  }
  
  // Get label count
  *numLabels = 256 * buf[6] + buf[7];
  if (*numLabels >= 100000) {
    fprintf(stderr, "Count (%d) is too high in %s\n", *numLabels, labelName);
    return 0;
  }
  
  // Allocate heap space for buffer
  *buffer = (unsigned char *) malloc(*numLabels * sizeof(unsigned char));
  
  // Actually do the read here
  read = fread(*buffer, sizeof(unsigned char), *numLabels, labelFile);
  if (read != *numLabels) {
    fprintf(stderr, "Error reading data in %s, %ld != %d\n", labelName, (long)read, *numLabels);
    return 0;
  }
  
  // All done!
  fclose(labelFile);
  return 1;
}

// Read all the images
static int readImages(char *imageName, int *numImages, int *rows, int *columns, unsigned char **buffer) {
  FILE *imageFile;
  unsigned char buf[16];
  size_t read;
  int size;
  
  // Open image file for reading
  imageFile = fopen(imageName, "r");
  if (imageFile == NULL) {
    fprintf(stderr, "Unable to read %s\n", imageName);
    return 0;
  }
  
  // Read header
  read = fread(buf, sizeof(unsigned char), 16, imageFile);
  if (read != 16 || buf[0] != 0 || buf[1] != 0 || buf[2] != 8 || buf[3] != 3) {
    fprintf(stderr, "Invalid header in file %s, read=%ld\n", imageName, (long)read);
    return 0;
  }
  
  // Get image count
  *numImages = 256 * buf[6] + buf[7];
  
  // Get image dimensions
  *rows = buf[11];
  *columns = buf[15];
  
  // Allocate space
  size = *rows * *columns * *numImages;
  *buffer = (unsigned char *) malloc (size * sizeof(unsigned char));
  
  // Do the massive read here
  read = fread(*buffer, sizeof(unsigned char), size, imageFile);
  if (read != size) {
    fprintf(stderr, "Error reading data in %s, %ld != %d\n", imageName, (long)read, size);
    return 0;
  }
  
  // Done!
  fclose(imageFile);
  return 1;
}

// Display an image to stdout
static void prtImage(int indx, unsigned char *labels, int rows, int columns, unsigned char *images) {
  int i, j;
  unsigned char *ptr;
  
  // Print header line
  printf("\nImage # %d is a %d", indx+1, labels[indx]);
  printf("\n      ");
  for (i=0; i<columns; i++) {
    printf(" %3d", i+1);
  }
  printf("\n");
  
  // Get to the start of the image (stored by rows, one byte per pixel)
  ptr = images + columns * rows * indx;
  
  for (j=0; j<rows; j++) {
    printf("  %2d: ", j+1);
    for (i=0; i<columns; i++) {
      if (*ptr == 0) {
        printf(" %3c", ' ');
      }
      else {
        printf(" %3d", *ptr);
      }
      ptr++;
    }
    printf("\n");
  }
}

// Small main function, manages all the real work
static int doit(char *name) {
  char imageName[100];
  char labelName[100];
  unsigned char *images;
  unsigned char *labels;
  int numLabels;
  int numImages;
  int rows, columns;
  int i;
  
  // Read all the labels
  strcpy(labelName, name);
  strcat(labelName, "-labels-idx1-ubyte");
  if (!readLabels(labelName, &numLabels, &labels)) return 0;
  
  // Read all the images
  strcpy(imageName, name);
  strcat(imageName, "-images-idx3-ubyte");
  if (!readImages(imageName, &numImages, &rows, &columns, &images)) return 0;

  for (i=0; i<5; i++) prtImage(i, labels, rows, columns, images);
  
  return 1;
}

// Main program, just pick up train or test
int main(int argc, char *argv[]) {
  char *name;
  
  if (argc != 2) {
    printf("Usage: reader train|t10k\n");
    exit(1);
  }
  if (strcmp(argv[1], "train") == 0 || strcmp(argv[1], "t10k") == 0) {
    name = argv[1];
  }
  else {
    printf("Usage: reader train|t10k\n");
    exit(2);
  }
  
  doit(name);
  return 0;
}
