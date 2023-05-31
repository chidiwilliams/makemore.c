#include "makemore.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/_types/_null.h>
#include <sys/_types/_size_t.h>
#include <sys/_types/_ssize_t.h>
#include <time.h>

void test_bigram() {
  double **bigram = bigram_init();

  {
    FILE *stream = fopen("names.txt", "r");
    if (stream == NULL) {
      exit(1);
    }

    size_t len = 0;
    char *line = NULL;
    ssize_t read;
    while ((read = getline(&line, &len, stream)) != -1) {
      bigram_add_word(bigram, line, read - 1);
    }

    free(line);
    fclose(stream);
  }

  bigram_normalize(bigram);

  bigram_print(bigram);

  // Sample from bigram
  const int num_samples = 10;
  for (int i = 0; i < num_samples; i++) {
    bigram_sample(bigram);
  }

  char *test_words[] = {"andrejq"};
  double num_test_words =
      (double)sizeof(test_words) / (double)sizeof(test_words[0]);
  double average_nll = bigram_average_nll(bigram, test_words, num_test_words);
  printf("nll/n = %f\n", average_nll);

  bigram_free(bigram);
}

void test_value() {
  Value *x1 = value_init_constant_with_label(2, "x1");
  Value *x2 = value_init_constant_with_label(0, "x2");

  Value *w1 = value_init_constant_with_label(-3, "w1");
  Value *w2 = value_init_constant_with_label(1, "w2");

  Value *b = value_init_constant_with_label(6.88113735870195432, "b");

  Value *result = value_tanh(
      value_add(value_add(value_times(x1, w1), value_times(x2, w2)), b));

  value_backward_tree(result);
  value_print_tree(result);

  value_free_tree(result);
}

void test_mlp() {
  srandom(time(NULL));

  Value *inputs[] = {value_init_constant(2), value_init_constant(3),
                     value_init_constant(-1)};
  int num_inputs = sizeof(inputs) / sizeof(inputs[0]);

  int layer_outputs[] = {2, 1};
  int num_layer_outputs = sizeof(layer_outputs) / sizeof(layer_outputs[0]);

  MLP *mlp = mlp_init(num_inputs, layer_outputs, num_layer_outputs);

  Value **results = mlp_apply(mlp, inputs);

  int num_results = layer_outputs[num_layer_outputs - 1];

  for (int i = 0; i < num_results; i++) {
    value_print_tree(results[i]);
  }

  for (int i = 0; i < num_results; i++) {
    value_free(results[i]);
  }

  mlp_free(mlp);
}

int main() {
  srand(0);
  test_bigram();
  return 0;
}
