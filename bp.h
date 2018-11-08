/*
Daniel Peek qer419
Michael Canas ohh135
CS3793 Assignment 06
11/8/2018
Based on code provided by Dr. O'Hara
*/

#ifndef BP_H
#define BP_H

#include <stdio.h>

// Because the input is a known size, we made the arrays all fixed size
// this simplifies the code in bp.c, fewer pointers and allocations needed
typedef struct backProp {
    double eta;                             // learning rate

    double biasBottom[28][28];              // bias for bottom weights
    double weightBottom[28][28][28][28];    // weights from input nodes(28x28) to hidden nodes(28x28)
    double biasTop[10];                     // bias for top weights
    double weightTop[28][28][10];           // weights from hidden nodes(28x28) to output nodes(10x1)

    double hidden[28][28];                  // hidden nodes
    double output[10];                      // output nodes
} backProp_t;

// Create the structure
extern backProp_t *createBP(double eta);

// Forward pass -- make a guess
extern int predictBP(backProp_t *bp, double input[28][28]);

// Got it wrong, adjust weights
extern void adjustWeightsBP(backProp_t *bp, double input[28][28], int actual);

#endif // BP_H

