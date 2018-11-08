/*
Daniel Peek qer419
Michael Canas ohh135
CS3793 Assignment 05
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

backProp_t *createBP(double eta) {
    int i, j, k, m;

    backProp_t *bp = (backProp_t *) calloc(1, sizeof(backProp_t));
    bp->eta = eta;

    for (i = 0; i < 28; i++){
        for (j = 0; j < 28; j++){
            for (k = 0; k < 28; k++){
                for (m = 0; m < 28; m++){
                    bp->weightBottom[i][j][k][m] = randWeight();
                }
            }
        }
    }

    for (i = 0; i < 28; i++){
        for (j = 0; j < 28; j++){
            for (k = 0; k < 10; k++){
                bp->weightTop[i][j][k] = randWeight();
            }
        }
    }

    for (i = 0; i < 28; i++){
        for (j = 0; j < 28; j++){
            bp->biasBottom[i][j] = randWeight();
        }
    }

    for (i = 0; i < 10; i++){
        bp->biasTop[i] = randWeight();
    }

    return bp;
}

// Feed forward values from inputs to hiddens to outputs
int predictBP(backProp_t *bp, double input[28][28]) {
    int i, j, k, m;
    double sum;

    // Calculate hidden values
    for (i = 0; i < 28; i++){
        for (j = 0; j < 28; j++){
            sum = 0.0;
            for (k = 0; k < 28; k++){
                for (m = 0; m < 28; m++){
                    //if ( abs((k + m)-(i + j))<= 2) {
                        sum += bp->weightBottom[i][j][k][m] * input[k][m];
                    //}
                }
            }
            sum += bp->biasBottom[i][j];
            bp->hidden[i][j] = 1.0 / (1.0 + exp(-sum));
        }
    }

    // Calculate output values
    for (i = 0; i < 28; i++){
        sum = 0.0;
        for (j = 0; j < 28; j++){
            for (k = 0; k < 10; k++){
                sum += bp->weightTop[i][j][k] * bp->hidden[i][j];
            }
        }
        sum += bp->biasTop[i];
        bp->output[i] = 1.0 / (1.0 + exp(-sum));
    }

    // Find highest output activation (class = i)
    for (k = 0; k < 10; k++) {
        if (k == 0 || bp->output[k] > bp->output[i]) {
            i = k;
        }
    }

    // Set caller's variable
    return i;
}

// Feed errors backwards through hiddens to inputs, by adjusting weights
void adjustWeightsBP(backProp_t *bp, double input[28][28], int actual) {
    int i, j, k, m, n;
    double sum;
    double delta[10];

    // Propagate the error backwards
    for (k = 0; k < 10; k++){
        sum = (k == actual) ? 1.0 : 0.0;
        sum -= bp->output[k];

        delta[k] = sum * bp->output[k] * (1 - bp->output[k]);

        for (i = 0; i < 28; i++){
            for (j = 0; j < 28; j++){
                bp->weightTop[i][j][k] += bp->eta * delta[k] * bp->hidden[i][j];
            }
        }

        bp->biasTop[k] += bp->eta * delta[k];
    }

    for (j = 0; j < 28; j++){
        for (k = 0; k < 28; k++){
            double d = 0.0;
            for (m = 0; m < 10; m++){
                d += bp->weightTop[j][k][m] * delta[m];
            }

            for (m = 0; m < 28; m++){
                for (n = 0; n < 28; n++){
                    //if ( abs((k + m)-(i + j))<= 2) {
                        bp->weightBottom[m][n][j][k] +=
                                bp->eta * bp->hidden[j][k] * (1 - bp->hidden[j][k]) * d * input[m][n];
                    //}
                }
            }

            bp->biasBottom[j][k] += bp->eta * d * bp->hidden[j][k] * (1 - bp->hidden[j][k]);
        }
    }
}
