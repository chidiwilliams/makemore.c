#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ALPHABET_SIZE 27

void print_bigram(double bigram[ALPHABET_SIZE][ALPHABET_SIZE]) {
  for (int i = 0; i < ALPHABET_SIZE; i++) {
    for (int j = 0; j < ALPHABET_SIZE; j++) {
      printf("%5f ", bigram[i][j]);
    }
    printf("\n");
  }
}

int sample_multinomial(double *values, int size) {
  double *cumulatives = (double *)malloc(size * sizeof(double));

  cumulatives[0] = values[0];
  for (int i = 1; i < size; i++) {
    cumulatives[i] = cumulatives[i - 1] + values[i];
  }

  // Get random number up to total
  double total = cumulatives[size - 1];
  double random_num = (float)rand() / RAND_MAX * total;

  // Get index of first cumulative value exceeding random number
  int index = 0;
  while (index < size && random_num >= cumulatives[index]) {
    index++;
  }

  free(cumulatives);
  return index;
}

#define char_to_index(char) (char - 'a' + 1)
#define index_to_char(index) ('a' + index - 1)

int main() {
  srand(0);
  double bigram[ALPHABET_SIZE][ALPHABET_SIZE] = {0};

  // Build bigram
  {
    FILE *stream;
    size_t len = 0;
    char *line = NULL;

    stream = fopen("names.txt", "r");
    if (stream == NULL) {
      exit(1);
    }

    while (getline(&line, &len, stream) != -1) {
      // Add start token
      bigram[0][char_to_index(line[0])] += 1;

      // Add characters
      int i;
      for (i = 1; line[i] != '\0'; i++) {
        bigram[char_to_index(line[i - 1])][char_to_index(line[i])] += 1;
      }

      // Add end token
      bigram[char_to_index(line[i - 2])][0] += 1;
    }

    free(line);
    fclose(stream);
  }

  // Add 1
  for (int i = 0; i < ALPHABET_SIZE; i++) {
    for (int j = 0; j < ALPHABET_SIZE; j++) {
      bigram[i][j] += 1;
    }
  }

  // Calculate probabilities for each row
  for (int i = 0; i < ALPHABET_SIZE; i++) {
    int total = 0;
    for (int j = 0; j < ALPHABET_SIZE; j++) {
      total += bigram[i][j];
    }
    for (int j = 0; j < ALPHABET_SIZE; j++) {
      bigram[i][j] /= total;
    }
  }

  print_bigram(bigram);

  // Sample from bigram
  const int NUM_SAMPLES = 10;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    int index = 0;
    while (1) {
      double *row = bigram[index];
      index = sample_multinomial(row, ALPHABET_SIZE);
      if (index == 0) {
        break;
      }
      printf("%c", index_to_char(index));
    }
    printf("\n");
  }

  // Calculate negative log likelihood
  {
    char *test = "andrejq";
    double log_likelihood = 0;
    double n = 0;
    {
      log_likelihood += log(bigram[0][char_to_index(test[0])]);
      int i;
      for (i = 1; test[i] != '\0'; i++) {
        log_likelihood +=
            log(bigram[char_to_index(test[i - 1])][char_to_index(test[i])]);
      }
      log_likelihood += log(bigram[char_to_index(test[i - 2])][0]);
      n += i + 1;
    }
    double negative_log_likelihood = -log_likelihood;
    printf("nll = %f\n", negative_log_likelihood);
    printf("nll/n = %f\n", negative_log_likelihood / n);
  }

  exit(0);
}
