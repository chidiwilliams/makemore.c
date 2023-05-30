#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/_types/_null.h>

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
  double *cumulatives = (double *)realloc(NULL, size * sizeof(double));

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

void run_bigram() {
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

  // Normalize probabilities for each row
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

typedef struct Value {
  double data;
  char *label;
  struct Value *leftChild;
  struct Value *rightChild;
} Value;

Value *init_value(double data, char *label) {
  Value *value = (Value *)realloc(NULL, sizeof(Value));
  value->data = data;
  value->label = label;
  value->leftChild = NULL;
  value->rightChild = NULL;
  return value;
}

Value *init_binary_value(double data, char *label, Value *leftChild,
                         Value *rightChild) {
  Value *value = init_value(data, label);
  value->leftChild = leftChild;
  value->rightChild = rightChild;
  return value;
}

Value *value_add(Value *value1, Value *value2) {
  Value *result =
      init_binary_value(value1->data + value2->data, "+", value1, value2);
  return result;
}

Value *valueTimes(Value *value1, Value *value2) {
  Value *result =
      init_binary_value(value1->data * value2->data, "*", value1, value2);
  return result;
}

void print_value(Value *value, int depth) {
  if (value == NULL) {
    return;
  }
  printf("[%4s | %f]\n", value->label, value->data);

  if (value->leftChild != NULL) {
    printf("%*s", (depth + 1) * 4, " ");
    print_value(value->leftChild, depth + 1);
  }

  if (value->rightChild != NULL) {
    printf("%*s", (depth + 1) * 4, " ");
    print_value(value->rightChild, depth + 1);
  }
}

void free_value(Value *value) {
  if (value == NULL) {
    return;
  }

  free_value(value->leftChild);
  free_value(value->rightChild);

  free(value);
}

int main() {
  Value *x1 = init_value(2.0, "x1");
  Value *x2 = init_value(0.0, "x2");

  Value *w1 = init_value(-3.0, "w1");
  Value *w2 = init_value(1.0, "w2");

  Value *b = init_value(6.88113735870195432, "b");

  Value *result =
      value_add(value_add(valueTimes(x1, w1), valueTimes(x2, w2)), b);

  print_value(result, 0);

  free_value(result);
}
