#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "bp.h"

// Random initial weight, from -1 to 1
double randWeight() {
  return ((2.0 * rand()) / INT_MAX - 1.0);
}

backProp_t *createBP(int nins, int nhiddens, int nouts, double eta) {
  int i, k;

  backProp_t *bp = (backProp_t *) calloc(1, sizeof(backProp_t));
  bp->inputs = nins;
  bp->hiddens = nhiddens;
  bp->classes = nouts;
  bp->eta = eta;

  bp->weightBottom = (double **) calloc(bp->inputs, sizeof(double *));
  for (i = 0; i < bp->inputs; i++) {
    bp->weightBottom[i] = (double *) calloc(bp->hiddens, sizeof(double));
    for (k = 0; k < bp->hiddens; k++) {
      bp->weightBottom[i][k] = randWeight();
    }
  }

  bp->weightTop = (double **) calloc(bp->hiddens, sizeof(double *));
  for (i = 0; i < bp->hiddens; i++) {
    bp->weightTop[i] = (double *) calloc(bp->classes, sizeof(double));
    for (k = 0; k < bp->classes; k++) {
      bp->weightTop[i][k] = randWeight();
    }
  }

  bp->biasBottom = (double *) calloc(bp->hiddens, sizeof(double));
  for (k = 0; k < bp->hiddens; k++) {
    bp->biasBottom[k] = randWeight();
  }

  bp->biasTop = (double *) calloc(bp->classes, sizeof(double));
  for (k = 0; k < bp->classes; k++) {
    bp->biasTop[k] = randWeight();
  }

  bp->hidden = (double *) calloc(bp->hiddens, sizeof(double));
  bp->output = (double *) calloc(bp->classes, sizeof(double));

  return bp;
}

// Feed forward values from inputs to hiddens to outputs
int predictBP(backProp_t *bp, double *sample, double *confidence) {
  int i, k;
  double sum;
  double nextMaxSum;

  // Calculate hidden values
  for (k = 0; k < bp->hiddens; k++) {
    sum = 0.0;
    for (i = 0; i < bp->inputs; i++) {
      sum += bp->weightBottom[i][k] * sample[i];
    }
    sum += bp->biasBottom[k];
    bp->hidden[k] = 1.0 / (1.0 + exp(-sum));    // Sigmoid (logistic)
  }

  // Calculate output values
  for (k = 0; k < bp->classes; k++) {
    sum = 0.0;
    for (i = 0; i < bp->hiddens; i++) {
      sum += bp->weightTop[i][k] * bp->hidden[i];
    }
    sum += bp->biasTop[k];
    bp->output[k] = 1.0 / (1.0 + exp(-sum));    // Sigmoid (logistic)
  }
  
  // Find highest output activation (class = i)
  for (k = 0; k < bp->classes; k++) {
    if (k == 0 || bp->output[k] > bp->output[i]) {
      i = k;
    }
  }

  // Find second largest number
  nextMaxSum = -1.0;
  for (k = 0; k < bp->classes; k++) {
    if (k != i) {
      if (nextMaxSum < 0.0 || bp->output[k] > nextMaxSum) {
        nextMaxSum = bp->output[k];
      }
    }
  }
  
  // Set caller's variable
  *confidence = bp->output[i] - nextMaxSum;
  return i;
}

// Feed errors backwards through hiddens to inputs, by adjusting weights
void adjustWeightsBP(backProp_t *bp, double *sample, int actual) {
  int i, j, k;
  double sum;
  double delta[bp->classes];

  // Propagate the error backwards
  for (k = 0; k < bp->classes; k++) {
    sum = (k == actual) ? 1.0 : 0.0;        // 1 if correct, else 0
    sum -= bp->output[k];                   // Predicted

    delta[k] = sum * bp->output[k] * (1 - bp->output[k]);   // Derivative of logistic (sigmoid)

    // Update weights from hiddens to outputs
    for (i = 0; i < bp->hiddens; i++) {
      bp->weightTop[i][k] += bp->eta * delta[k] * bp->hidden[i];
    }

    // Update bias from hiddens to outputs
    bp->biasTop[k] += bp->eta * delta[k];
  }

  for (j = 0; j < bp->hiddens; j++) {
    double d = 0;
    for (k = 0; k < bp->classes; k++) {
      d += bp->weightTop[j][k] * delta[k];
    }

    // Update weights from inputs to hiddens
    for (i = 0; i < bp->inputs; i++) {
      bp->weightBottom[i][j] += bp->eta * bp->hidden[j] * (1 - bp->hidden[j]) * d * sample[i];
    }

    // Update bias from inputs to hiddens
    bp->biasBottom[j] += bp->eta * d * bp->hidden[j] * (1 - bp->hidden[j]);
  }
}
