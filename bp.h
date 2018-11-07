/*
Daniel Peek qer419
Michael Canas ohh135
CS3793 Assignment 05
11/8/2018
Using code provided by Dr. O'Hara
*/

#ifndef BP_H
#define BP_H

typedef struct backProp {
    int inputs;
    int hiddens;
    int classes;

    double eta;

    double *biasBottom;
    double **weightBottom;
    double *biasTop;
    double **weightTop;

    double *hidden;
    double *output;
} backProp_t;

// Create the structure
extern backProp_t *createBP(int nins, int nhiddens, int nouts, double eta);

// Print the whoel network
extern void printBP(FILE *out, backProp_t *bp);

// Forward pass -- make a guess
extern int predictBP(backProp_t *bp, double *sample, double *confidence);

// Print the feed forward pass
extern void prtPrediction(FILE *out, backProp_t *bp, double *sample);

// Got it wrong, adjust weights
extern void adjustWeightsBP(backProp_t *bp, double *sample, int actual);

#endif // BP_H

