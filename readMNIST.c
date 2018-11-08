// Reference http://yann.lecun.com/exdb/mnist/ as of Oct 23, 2018
//
// This reads the MNIST labelled digit database and prints the first few samples
//
// To compile: gcc readMNIST.c -o /tmp/readMNIST
//
// To run: /tmp/readMNIST train
//     or: /tmp/readMNIST t10k
//

/*
Daniel Peek qer419
Michael Canas ohh135
CS3793 Assignment 05
11/8/2018
Based on code provided by Dr. O'Hara
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bp.h"

int readWeightsFromFile;
int training;

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
        fprintf(stderr, "Invalid header in file %s, read=%ld\n", labelName, (long) read);
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
        fprintf(stderr, "Error reading data in %s, %ld != %d\n", labelName, (long) read, *numLabels);
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
        fprintf(stderr, "Invalid header in file %s, read=%ld\n", imageName, (long) read);
        return 0;
    }

    // Get image count
    *numImages = 256 * buf[6] + buf[7];

    // Get image dimensions
    *rows = buf[11];
    *columns = buf[15];

    // Allocate space
    size = *rows * *columns * *numImages;
    *buffer = (unsigned char *) malloc(size * sizeof(unsigned char));

    // Do the massive read here
    read = fread(*buffer, sizeof(unsigned char), size, imageFile);
    if (read != size) {
        fprintf(stderr, "Error reading data in %s, %ld != %d\n", imageName, (long) read, size);
        return 0;
    }

    // Done!
    fclose(imageFile);
    return 1;
}

// Display an image to stdout
static void prtImage(int indx, unsigned char *labels, int rows, int columns, unsigned char *images, int input[28][28]) {
    int i, j;
    unsigned char *ptr;

    // Print header line
    // printf("\nImage # %d is a %d", indx + 1, labels[indx]);
    //printf("\n      ");
    //for (i = 0; i < columns; i++) {
        //printf(" %3d", i + 1);
    //}
    //printf("\n");

    // Get to the start of the image (stored by rows, one byte per pixel)
    ptr = images + columns * rows * indx;

    for (i = 0; i < rows; i++) {
        //printf("  %2d: ", j + 1);
        for (j = 0; j < columns; j++) {
            /*
            if (*ptr == 0) {
                printf(" %3c", ' ');
            } else {
                printf(" %3d", *ptr);
            }*/
            input[i][j] = *ptr;
            ptr++;
        }
        //printf("\n");
    }
    /*
    for (int i = 0; i < 28; i++){
        for (int j = 0; j < 28; j++){
            if (input[i][j] == 0){
                printf("%3c", ' ');
            }
            else {
                printf("%3d", input[i][j]);
            }
            if (j == 27){
                printf("\n");
            }
        }
    }
    */
}

// Small main function, manages all the real work
static int doit(char *name) {
    char imageName[100];
    char labelName[100];
    int input[28][28];
    double float_input[28][28];
    unsigned char *images;
    unsigned char *labels;
    int numLabels;
    int numImages;
    int rows, columns;
    int run;
    int guess;
    int correct = 0;
    int total = 0;
    FILE *bottomWeights;
    FILE *topWeights;
    FILE *bottomBiases;
    FILE *topBiases;

    backProp_t *backprop = createBP(.001);

    if (readWeightsFromFile){
        bottomWeights = fopen("bottom_weights", "rb");
        topWeights = fopen("top_weights", "rb");
        bottomBiases = fopen("bottom_biases", "rb");
        topBiases = fopen("top_biases", "rb");

        fread(backprop->weightBottom, sizeof(double), sizeof(backprop->weightBottom), bottomWeights);
        fread(backprop->weightTop, sizeof(double), sizeof(backprop->weightTop), topWeights);
        fread(backprop->biasBottom, sizeof(double), sizeof(backprop->biasBottom), bottomBiases);
        fread(backprop->biasTop, sizeof(double), sizeof(backprop->biasTop), topBiases);

        fclose(bottomWeights);
        fclose(topWeights);
        fclose(bottomBiases);
        fclose(topBiases);
    }

    srand(time(0));


    // Read all the labels
    strcpy(labelName, name);
    strcat(labelName, "-labels-idx1-ubyte");
    if (!readLabels(labelName, &numLabels, &labels)) return 0;

    // Read all the images
    strcpy(imageName, name);
    strcat(imageName, "-images-idx3-ubyte");
    if (!readImages(imageName, &numImages, &rows, &columns, &images)) return 0;

    for (run = 0; run < 480000; run++) {
        int index = (rand() % 60000);
      //  if ((int) labels[index] == 4) {
        prtImage(index, labels, rows, columns, images, input);
        for (int x = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                float_input[x][y] = (double) input[x][y];
            }
        }
        guess = predictBP(backprop, float_input);
        if (guess == (int) labels[index]) {
            correct++;
            total++;
        } else {
            total++;
        }
        if (training) {
            adjustWeightsBP(backprop, float_input, (int) labels[index]);
        }

        //printf("i = %d\n", i);

        if (run % 1000 == 0) {
            printf("Total accuracy at try %d: %lf\n", run, (double) correct / (double) total);
        }
     //   }
     //   else{
     //       run -= 1;
     //   }
    }

    printf("\nWrite weights to file? (Y/N): ");
    char saveToFile = (char) getchar();

    if (saveToFile == 'Y' || saveToFile == 'y'){
        bottomWeights = fopen("bottom_weights", "wb");
        topWeights = fopen("top_weights", "wb");
        bottomBiases = fopen("bottom_biases", "wb");
        topBiases = fopen("top_biases", "wb");

        fwrite(backprop->weightBottom, sizeof(double), sizeof(backprop->weightBottom), bottomWeights);
        fwrite(backprop->weightTop, sizeof(double), sizeof(backprop->weightTop), topWeights);
        fwrite(backprop->biasBottom, sizeof(double), sizeof(backprop->biasBottom), bottomBiases);
        fwrite(backprop->biasTop, sizeof(double), sizeof(backprop->biasTop), topBiases);

        fclose(bottomWeights);
        fclose(topWeights);
        fclose(bottomBiases);
        fclose(topBiases);

    }

    return 1;
}

// Main program, just pick up train or test
int main(int argc, char *argv[]) {
    char *name;

    if (argc != 3) {
        printf("Usage: reader train|t10k 1|0\n");
        exit(1);
    }
    if (strcmp(argv[1], "train") == 0 || strcmp(argv[1], "t10k") == 0) {
        name = argv[1];
        if (strcmp(name, "train") == 0){
            training = 1;
        }
        else{
            training = 0;
        }
    } else {
        printf("Usage: reader train|t10k 1|0\n");
        exit(2);
    }
    readWeightsFromFile = atoi(argv[2]);
    if (readWeightsFromFile != 1 && readWeightsFromFile != 0){
        printf("Usage: reader train|t10k 1|0\n");
        exit(3);
    }

    doit(name);
    return 0;
}
