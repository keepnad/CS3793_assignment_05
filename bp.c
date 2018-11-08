/*
Daniel Peek qer419
Michael Canas ohh135
CS3793 Assignment 06
11/8/2018
Based on code provided by Dr. O'Hara
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include "bp.h"

// Random initial weight, from -1 to 1
double randWeight() {
    return ((2.0 * rand()) / INT_MAX - 1.0);
}

// fill in random initial values for all weights and biases
backProp_t *createBP(double eta) {
    int i, j, k, m;

    backProp_t *bp = (backProp_t *) calloc(1, sizeof(backProp_t));
    bp->eta = eta;

    for (i = 0; i < 28; i++) {
        for (j = 0; j < 28; j++) {
            for (k = 0; k < 28; k++) {
                for (m = 0; m < 28; m++) {
                    bp->weightBottom[i][j][k][m] = randWeight();
                }
            }
        }
    }

    for (i = 0; i < 28; i++) {
        for (j = 0; j < 28; j++) {
            for (k = 0; k < 10; k++) {
                bp->weightTop[i][j][k] = randWeight();
            }
        }
    }

    for (i = 0; i < 28; i++) {
        for (j = 0; j < 28; j++) {
            bp->biasBottom[i][j] = randWeight();
        }
    }

    for (i = 0; i < 10; i++) {
        bp->biasTop[i] = randWeight();
        bp->output[i] = 0;
    }

    return bp;
}

// Feed forward values from inputs to hiddens to outputs
int predictBP(backProp_t *bp, double input[28][28]) {
    int i, j, k, m;
    double sum;

    // Calculate hidden values
    for (i = 0; i < 28; i++) {
        for (j = 0; j < 28; j++) {
            sum = 0.0;

            // this is the partial connectivity, k goes from i - 2 to i + 2
            // m goes from j - 2 to j + 2, with exceptions made to keep them in the array bounds,
            // meaning that each of the hidden nodes only connects to the "nearby" input nodes
            for (k = ((i > 2) ? i - 2 : 0); k < ((i < 26) ? i + 2 : 28); k++) {
                for (m = ((j > 2) ? j - 2 : 0); m < ((j < 26) ? j + 2 : 28); m++) {
                    sum += bp->weightBottom[i][j][k][m] * input[i][j];
                }
            }
            sum += bp->biasBottom[i][j];
            bp->hidden[i][j] = 1.0 / (1.0 + exp(-sum)); // sigmoid
        }
    }

    // Calculate output values
    for (i = 0; i < 10; i++) {
        sum = 0.0;
        for (j = 0; j < 28; j++) {
            for (k = 0; k < 28; k++) {
                sum += bp->weightTop[j][k][i] * bp->hidden[j][k];
            }
        }
        sum += bp->biasTop[i];
        bp->output[i] = 1.0 / (1.0 + exp(-sum)); // sigmoid
    }

    // Find highest output activation (class = i)
    for (k = 0; k < 10; k++) {
        if (k == 0 || bp->output[k] > bp->output[i]) {
            i = k;
        }
    }

    // Return which number the neural network thinks is depicted
    return i;
}

// Feed errors backwards through hidden nodes to inputs, by adjusting weights
void adjustWeightsBP(backProp_t *bp, double input[28][28], int actual) {
    int i, j, k, m, n;
    double sum;
    double delta[10];

    // Propagate the error backwards
    for (k = 0; k < 10; k++) {
        sum = (k == actual) ? 1.0 : 0.0;
        sum -= bp->output[k];

        delta[k] = sum * bp->output[k] * (1 - bp->output[k]);

        for (i = 0; i < 28; i++) {
            for (j = 0; j < 28; j++) {
                bp->weightTop[i][j][k] += bp->eta * delta[k] * bp->hidden[i][j];
            }
        }

        bp->biasTop[k] += bp->eta * delta[k];
    }

    for (j = 0; j < 28; j++) {
        for (k = 0; k < 28; k++) {
            double d = 0.0;
            for (m = 0; m < 10; m++) {
                d += bp->weightTop[j][k][m] * delta[m];
            }

            // this is the other part that is partially connected. As in predictBP,
            //  each hidden node only gets affected by the nearby input nodes
            for (m = ((j > 2) ? j - 2 : 0); m < ((j < 26) ? j + 2 : 28); m++) {
                for ((n = (k > 2) ? k - 2 : 0); n < ((k < 26) ? k + 2 : 28); n++) {
                    bp->weightBottom[j][k][m][n] +=
                            bp->eta * bp->hidden[m][n] * (1 - bp->hidden[m][n]) * d * input[j][k];
                }
            }

            bp->biasBottom[j][k] += bp->eta * d * bp->hidden[j][k] * (1 - bp->hidden[j][k]); // derivative of sigmoid
        }
    }
}
