/*
Daniel Peek qer419
Michael Canas ohh135
CS3793 Assignment 05
11/8/2018
Using code provided by Dr. O'Hara
*/

#include <stdio.h>

#include "bp.h"
/*
void printBP(FILE *out, backProp_t *bp) {
    int i, k;

    fprintf(out, "\nInputs=%s Hiddens=%s Outputs=%s\n\n", "28 x 28\0", "28 x 28\0", "10\0");

    // Show all the input -> hidden weights
    for (i = 0; i < 28; i++) {
        fprintf(out, "weight input->hidden[%d][0:%d] =", i, bp->hiddens);
        for (k = 0; k < bp->hiddens; k++) {
            fprintf(out, " %6.3f", bp->weightBottom[i][k]);
        }
        fprintf(out, "\n");
    }
    fprintf(out, "\n");

    // Show all the hidden -> output weights
    for (i = 0; i < bp->hiddens; i++) {
        fprintf(out, "weight hidden->output[%d][0:%d] =", i, bp->classes);
        for (k = 0; k < bp->classes; k++) {
            fprintf(out, " %6.3f", bp->weightTop[i][k]);
        }
        fprintf(out, "\n");
    }
    fprintf(out, "\n");

    // Show all the hidden biases
    fprintf(out, "bias hiddens[0:%d] =", bp->hiddens);
    for (k = 0; k < bp->hiddens; k++) {
        fprintf(out, " %6.3f", bp->biasBottom[k]);
    }
    fprintf(out, "\n\n");

    // Show all the output biases
    fprintf(out, "bias output[0:%d] =", bp->classes);
    for (k = 0; k < bp->classes; k++) {
        fprintf(out, " %6.3f", bp->biasTop[k]);
    }
    fprintf(out, "\n\n");
}

void prtPrediction(FILE *out, backProp_t *bp, double *sample) {
    int i, k;

    // Calculate hidden values
    for (k = 0; k < bp->hiddens; k++) {
        fprintf(out, "Hidden[%d] = sigmoid(", k);

        for (i = 0; i < bp->inputs; i++) {
            if (i > 0) fprintf(out, " + ");
            fprintf(out, "%3.1f*%6.3f", sample[i], bp->weightBottom[i][k]);
        }
        fprintf(out, " + %6.3f", bp->biasBottom[k]);
        fprintf(out, ") = %6.3f\n", bp->hidden[k]);
    }
    fprintf(out, "\n");

    // Calculate output values
    for (k = 0; k < bp->classes; k++) {
        fprintf(out, "Output[%d] = sigmoid(", k);

        for (i = 0; i < bp->hiddens; i++) {
            if (out && i > 0) fprintf(out, " + ");
            fprintf(out, "%6.3f*%6.3f", bp->hidden[i], bp->weightTop[i][k]);
        }
        fprintf(out, " + %6.3f", bp->biasTop[k]);
        fprintf(out, ") = %6.3f\n", bp->output[k]);
    }
    fprintf(out, "\n");
}
*/