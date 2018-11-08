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
CS3793 Assignment 06
11/8/2018
Based on code provided by Dr. O'Hara
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bp.h"

// Global variable to track if the weights are being read in from files
int readWeightsFromFile;

// Global variable to track if the program is running in training or testing mode
int training;

// read each weight and bias array from a file
static void readWeights(backProp_t *bp) {
    FILE *bottomWeights = fopen("bottom_weights", "rb");
    FILE *topWeights = fopen("top_weights", "rb");
    FILE *bottomBiases = fopen("bottom_biases", "rb");
    FILE *topBiases = fopen("top_biases", "rb");

    fread(bp->weightBottom, sizeof(double), sizeof(bp->weightBottom), bottomWeights);
    fread(bp->weightTop, sizeof(double), sizeof(bp->weightTop), topWeights);
    fread(bp->biasBottom, sizeof(double), sizeof(bp->biasBottom), bottomBiases);
    fread(bp->biasTop, sizeof(double), sizeof(bp->biasTop), topBiases);

    fclose(bottomWeights);
    fclose(topWeights);
    fclose(bottomBiases);
    fclose(topBiases);
}

// write each weight and bias array to a file
static void writeWeights(backProp_t *bp) {
    FILE *bottomWeights = fopen("bottom_weights", "wb");
    FILE *topWeights = fopen("top_weights", "wb");
    FILE *bottomBiases = fopen("bottom_biases", "wb");
    FILE *topBiases = fopen("top_biases", "wb");

    fwrite(bp->weightBottom, sizeof(double), sizeof(bp->weightBottom), bottomWeights);
    fwrite(bp->weightTop, sizeof(double), sizeof(bp->weightTop), topWeights);
    fwrite(bp->biasBottom, sizeof(double), sizeof(bp->biasBottom), bottomBiases);
    fwrite(bp->biasTop, sizeof(double), sizeof(bp->biasTop), topBiases);

    fclose(bottomWeights);
    fclose(topWeights);
    fclose(bottomBiases);
    fclose(topBiases);
}

// Read all the labels - not modified from provided code
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

// Read all the images - not modified from provided code
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

// Store the image into a 2D double array
static void storeImage(int index, int rows, int columns, unsigned char *images, double input[28][28]) {
    int i, j;
    unsigned char *ptr;

    ptr = images + columns * rows * index;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            input[i][j] = (double) *ptr;
            ptr++;
        }
    }
}

// Small main function, manages all the real work
static int doit(char *name) {

    // These variables are from the unmodified file
    char imageName[100];
    char labelName[100];
    unsigned char *images;
    unsigned char *labels;
    int numLabels;
    int numImages;
    int rows, columns;

    // We added these variables
    double input[28][28];   // 2D array for the 28 x 28 input images
    int run;                // counter variable for main loop
    int numRuns;            // total number of runs to execute
    int guess;              // result of the prediction
    int correct = 0;        // totals for how many numbers have been guessed overall
    int total = 0;          // and how many have been guessed correctly
    int displayFreq;        // display the accuracy when (run % displayFreq) == 0

    //create the backprop object
    backProp_t *backprop = createBP(.001);

    // read weights in from file, if desired
    //required for testing mode
    if (readWeightsFromFile) {
        readWeights(backprop);
    }

    // Read all the labels
    strcpy(labelName, name);
    strcat(labelName, "-labels-idx1-ubyte");
    if (!readLabels(labelName, &numLabels, &labels)) return 0;

    // Read all the images
    strcpy(imageName, name);
    strcat(imageName, "-images-idx3-ubyte");
    if (!readImages(imageName, &numImages, &rows, &columns, &images)) return 0;

    // if training, do more runs and display accuracy less often
    if (training) {
        numRuns = 1000000;
        displayFreq = 1000;
    } else {
        numRuns = numImages;
        displayFreq = 500;
    }

    srand(time(0));

    // main loop
    for (run = 0; run < numRuns; run++) {
        int index;
        // if training, go to random indices
        if (training) {
            index = (rand() % 60000);
        } else {
            index = run;
        }
        // get the image into the input array, then convert to double
        storeImage(index, rows, columns, images, input);

        // predict what number the image is
        guess = predictBP(backprop, input);
        if (guess == (int) labels[index]) {
            correct++;
            total++;
        } else {
            total++;
        }

        // if in training, adjust the weights to improve the network
        if (training) {
            adjustWeightsBP(backprop, input, (int) labels[index]);
        }

        // print accuracy
        if (run % displayFreq == 0 || run == numRuns - 1) {
            printf("Total accuracy at try %d: %.03lf%%\n", run, ((double) correct / (double) total) * 100.0);
        }
    }

    // if the net is training, the weights can be saved for later use
    if (training) {
        printf("\nWrite weights to file? (Y/N): ");
        char saveToFile = (char) getchar();

        if (saveToFile == 'Y' || saveToFile == 'y') {
            writeWeights(backprop);
        }
    }
    return 1;
}

// Main program, pick train or test, and whether or not to read weights from file
int main(int argc, char *argv[]) {
    char *name;
    // check for correct number of args
    if (argc != 3) {
        printf("Usage: CS3793_assignment_06 <train|t10k> <0|1>\n");
        printf("\ttrain: training mode, for training the neural network\n");
        printf("\tt10k:  testing mode, for testing a trained neural network\n");
        printf("\t0: Start with a new, random network\n\t1: Open saved existing weights\n");
        exit(1);
    }
    // if first argument is not correct
    if (strcmp(argv[1], "train") == 0 || strcmp(argv[1], "t10k") == 0) {
        name = argv[1];
        if (strcmp(name, "train") == 0) {
            training = 1;
        } else {
            training = 0;
        }
    } else {
        printf("Usage: CS3793_assignment_06 <train|t10k> <0|1>\n");
        printf("\ttrain: training mode, for training the neural network\n");
        printf("\tt10k:  testing mode, for testing a trained neural network\n");
        printf("\t0: Start with a new, random network\n\t1: Open saved existing weights\n");
        exit(2);
    }
    // if second argument is not correct
    readWeightsFromFile = (int) strtol(argv[2], NULL, 10);
    if (readWeightsFromFile != 1 && readWeightsFromFile != 0) {
        printf("Usage: CS3793_assignment_06 <train|t10k> <0|1>\n");
        printf("\ttrain: training mode, for training the neural network\n");
        printf("\tt10k:  testing mode, for testing a trained neural network\n");
        printf("\t0: Start with a new, random network\n\t1: Open saved existing weights\n");
        exit(3);
    }
    // if attempting to enter testing mode and not supply weights
    if (readWeightsFromFile == 0 && training == 0) {
        printf("Usage: CS3793_assignment_06 <train|t10k> <0|1>\n");
        printf("\t>>>> Testing mode requires use of pre-made weights. <<<<\n");
        printf("\ttrain: training mode, for training the neural network\n");
        printf("\tt10k:  testing mode, for testing a trained neural network\n");
        printf("\t0: Start with a new, random network\n\t1: Open saved existing weights\n");
        exit(4);
    }

    // run the thing
    doit(name);
    return 0;
}
