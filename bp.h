/*
Daniel Peek qer419
Michael Canas ohh135
CS3793 Assignment 05
11/8/2018
Based on code provided by Dr. O'Hara
*/

#ifndef BP_H
#define BP_H
#include <stdio.h>

typedef struct backProp {
    double eta;

    double biasBottom[28][28]; // bias for bottom weights
    double weightBottom[28][28][28][28]; // weights from input to hidden nodes
    double biasTop[10]; // bias for top weights
    double weightTop[28][28][10]; // weights from hidden to output nodes

    double hidden[28][28]; // hidden nodes
    double output[10]; // output nodes
} backProp_t;

// Create the structure
extern backProp_t *createBP(double eta);

// Print the whole network
extern void printBP(FILE *out, backProp_t *bp);

// Forward pass -- make a guess
extern int predictBP(backProp_t *bp, double input[28][28]);

// Print the feed forward pass
extern void prtPrediction(FILE *out, backProp_t *bp, double input[28][28]);

// Got it wrong, adjust weights
extern void adjustWeightsBP(backProp_t *bp, double input[28][28], int actual);

#endif // BP_H

