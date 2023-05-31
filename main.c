#include <alloca.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/_types/_null.h>
#include <sys/_types/_size_t.h>
#include <time.h>

#define ALPHABET_SIZE 27

void *allocate(size_t size) {
  void *result = malloc(size);
  if (result == NULL) {
    exit(1);
  }
  return result;
}

void bigram_print(double bigram[ALPHABET_SIZE][ALPHABET_SIZE]) {
  for (int i = 0; i < ALPHABET_SIZE; i++) {
    for (int j = 0; j < ALPHABET_SIZE; j++) {
      printf("%5f ", bigram[i][j]);
    }
    printf("\n");
  }
}

int sample_multinomial(double *values, int size) {
  double *cumulatives = (double *)allocate(size * sizeof(double));

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

#define CHAR_TO_INDEX(char) (char - 'a' + 1)
#define INDEX_TO_CHAR(index) ('a' + index - 1)

void test_bigram() {
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
      bigram[0][CHAR_TO_INDEX(line[0])] += 1;

      // Add characters
      int i;
      for (i = 1; line[i] != '\0'; i++) {
        bigram[CHAR_TO_INDEX(line[i - 1])][CHAR_TO_INDEX(line[i])] += 1;
      }

      // Add end token
      bigram[CHAR_TO_INDEX(line[i - 2])][0] += 1;
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

  bigram_print(bigram);

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
      printf("%c", INDEX_TO_CHAR(index));
    }
    printf("\n");
  }

  // Calculate negative log likelihood
  {
    char *test = "andrejq";
    double log_likelihood = 0;
    double n = 0;
    {
      log_likelihood += log(bigram[0][CHAR_TO_INDEX(test[0])]);
      int i;
      for (i = 1; test[i] != '\0'; i++) {
        log_likelihood +=
            log(bigram[CHAR_TO_INDEX(test[i - 1])][CHAR_TO_INDEX(test[i])]);
      }
      log_likelihood += log(bigram[CHAR_TO_INDEX(test[i - 2])][0]);
      n += i + 1;
    }
    double negative_log_likelihood = -log_likelihood;
    printf("nll = %f\n", negative_log_likelihood);
    printf("nll/n = %f\n", negative_log_likelihood / n);
  }

  exit(0);
}

enum ValueType { CONSTANT, ADD, MULTIPLY, TANH };

typedef struct Value {
  enum ValueType type;
  double data;
  double grad;
  char *label;
  struct Value *left_child;
  struct Value *right_child;
} Value;

Value *value_init(double data, enum ValueType type) {
  Value *value = (Value *)allocate(sizeof(Value));
  value->data = data;
  value->label = NULL;
  value->grad = 0.0;
  value->type = type;
  value->left_child = NULL;
  value->right_child = NULL;
  return value;
}

Value *value_init_constant(double data) { return value_init(data, CONSTANT); }

Value *value_init_constant_with_label(double data, char *label) {
  Value *value = value_init_constant(data);
  value->label = label;
  return value;
}

Value *value_init_binary(double data, enum ValueType type, Value *leftChild,
                         Value *rightChild) {
  Value *value = value_init(data, type);
  value->left_child = leftChild;
  value->right_child = rightChild;
  return value;
}

Value *value_init_unary(double data, enum ValueType type, Value *child) {
  Value *value = value_init(data, type);
  value->left_child = child;
  return value;
}

Value *value_add(Value *value1, Value *value2) {
  Value *result =
      value_init_binary(value1->data + value2->data, ADD, value1, value2);
  return result;
}

Value *value_times(Value *value1, Value *value2) {
  Value *result =
      value_init_binary(value1->data * value2->data, MULTIPLY, value1, value2);
  return result;
}

Value *value_tanh(Value *value) {
  Value *result = value_init_unary(tanh(value->data), TANH, value);
  return result;
}

void value_backward(Value *value) {
  switch (value->type) {
  case CONSTANT: {
    break;
  }
  case ADD: {
    value->left_child->grad = 1.0 * value->grad;
    value->right_child->grad = 1.0 * value->grad;
    break;
  }
  case MULTIPLY: {
    value->left_child->grad = value->right_child->data * value->grad;
    value->right_child->grad = value->left_child->data * value->grad;
    break;
  }
  case TANH: {
    double d = value->left_child->data;
    double t = (exp(2 * d) - 1) / (exp(2 * d) + 1);
    value->left_child->grad = (1 - pow(t, 2)) * value->grad;
    break;
  }
  }
}

void value_print(Value *value) {
  char *label;
  switch (value->type) {
  case CONSTANT:
    label = value->label;
    break;
  case ADD:
    label = "+";
    break;
  case MULTIPLY:
    label = "*";
    break;
  case TANH:
    label = "tanh";
    break;
  }

  printf("[%4s | %f | %f]\n", label, value->data, value->grad);
}

void value_print_tree_at_depth(Value *value, int depth) {
  if (value == NULL) {
    return;
  }

  value_print(value);

  if (value->left_child != NULL) {
    printf("%*s", (depth + 1) * 4, " ");
    value_print_tree_at_depth(value->left_child, depth + 1);
  }

  if (value->right_child != NULL) {
    printf("%*s", (depth + 1) * 4, " ");
    value_print_tree_at_depth(value->right_child, depth + 1);
  }
}

void value_print_tree(Value *value) { value_print_tree_at_depth(value, 0); }

typedef struct ValueItem {
  Value *value;
  struct ValueItem *next;
} ValueItem;

ValueItem *valueitem_init(Value *value) {
  ValueItem *item = (ValueItem *)allocate(sizeof(ValueItem));
  item->value = value;
  item->next = NULL;
  return item;
}

void sort_topological(ValueItem **sorted, ValueItem **visited, Value *value) {
  if (value == NULL) {
    return;
  }

  if (*visited != NULL) {
    ValueItem *current = *visited;
    while (current->next != NULL) {
      if (current->value == value) {
        return;
      }
      current = current->next;
    }
  }

  ValueItem *visitedItem = valueitem_init(value);
  if (*visited == NULL) {
    *visited = visitedItem;
  } else {
    ValueItem *current = *visited;
    while (current->next != NULL) {
      current = current->next;
    }
    current->next = visitedItem;
  }

  sort_topological(sorted, visited, value->left_child);
  sort_topological(sorted, visited, value->right_child);

  ValueItem *sortedItem = valueitem_init(value);
  if (*sorted == NULL) {
    *sorted = sortedItem;
  } else {
    sortedItem->next = *sorted;
    *sorted = sortedItem;
  }
}

void valueitem_free(ValueItem *valueItem) {
  if (valueItem == NULL) {
    return;
  }
  free(valueItem->next);
  free(valueItem);
}

void value_backward_tree(Value *value) {
  ValueItem *sorted = NULL;
  ValueItem *visited = NULL;

  sort_topological(&sorted, &visited, value);

  value->grad = 1;

  for (ValueItem *current = sorted; current != NULL; current = current->next) {
    value_backward(current->value);
  }

  valueitem_free(sorted);
  valueitem_free(visited);
}

void value_free_tree(Value *value) {
  if (value == NULL) {
    return;
  }

  value_free_tree(value->left_child);
  value_free_tree(value->right_child);

  free(value);
}

void value_free(Value *value) {
  if (value == NULL) {
    return;
  }

  free(value);
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

typedef struct Neuron {
  int num_inputs;
  Value **w;
  Value *b;
} Neuron;

#define RANDOM_WEIGHT() ((double)random() / (double)RAND_MAX)

Neuron *neuron_init(int num_inputs) {
  Neuron *neuron = (Neuron *)allocate(sizeof(Neuron));
  neuron->num_inputs = num_inputs;
  neuron->b = value_init_constant(RANDOM_WEIGHT());

  Value **w = (Value **)allocate(num_inputs * sizeof(Value));
  for (int i = 0; i < num_inputs; i++) {
    w[i] = value_init_constant(RANDOM_WEIGHT());
  }
  neuron->w = w;
  return neuron;
}

void neuron_free(Neuron *neuron) {
  for (int i = 0; i < neuron->num_inputs; i++) {
    value_free_tree(neuron->w[i]);
  }
  free(neuron->w);
  value_free_tree(neuron->b);
  free(neuron);
}

void neuron_print(Neuron *neuron) {
  for (int i = 0; i < neuron->num_inputs; i++) {
    printf("%4d: ", i);
    value_print(neuron->w[i]);
  }
  printf("%4s: ", "b");
  value_print(neuron->b);
}

Value *neuron_apply(Neuron *neuron, Value **inputs) {
  Value *activation = value_times(neuron->w[0], inputs[0]);
  for (int i = 1; i < neuron->num_inputs; i++) {
    activation = value_add(activation, value_times(neuron->w[i], inputs[i]));
  }
  activation = value_add(activation, neuron->b);
  return value_tanh(activation);
}

typedef struct Layer {
  int num_inputs;
  int num_outputs;
  Neuron **neurons;
} Layer;

Layer *layer_init(int num_inputs, int num_outputs) {
  Layer *layer = (Layer *)allocate(sizeof(Layer));
  layer->num_inputs = num_inputs;
  layer->num_outputs = num_outputs;
  layer->neurons = (Neuron **)allocate(num_outputs * sizeof(Neuron *));
  for (int i = 0; i < num_outputs; i++) {
    layer->neurons[i] = neuron_init(num_inputs);
  }
  return layer;
}

void layer_free(Layer *layer) {
  for (int i = 0; i < layer->num_outputs; i++) {
    neuron_free(layer->neurons[i]);
  }
  free(layer->neurons);
  free(layer);
}

Value **layer_apply(Layer *layer, Value **inputs) {
  Value **outputs = (Value **)allocate(layer->num_outputs * sizeof(Value *));
  for (int i = 0; i < layer->num_outputs; i++) {
    outputs[i] = neuron_apply(layer->neurons[i], inputs);
  }
  return outputs;
}

typedef struct MLP {
  int num_layers;
  Layer **layers;
} MLP;

MLP *mlp_init(int num_inputs, int *layer_outputs, int num_layer_outputs) {
  MLP *mlp = (MLP *)allocate(sizeof(MLP));
  mlp->num_layers = num_layer_outputs;
  mlp->layers = (Layer **)allocate(num_layer_outputs * sizeof(Layer *));
  for (int i = 0; i < num_layer_outputs; i++) {
    if (i == 0) {
      mlp->layers[i] = layer_init(num_inputs, layer_outputs[0]);
    } else {
      mlp->layers[i] = layer_init(layer_outputs[i - 1], layer_outputs[i]);
    }
  }
  return mlp;
}

Value **mlp_apply(MLP *mlp, Value **inputs) {
  Value **results = inputs;
  for (int i = 0; i < mlp->num_layers; i++) {
    results = layer_apply(mlp->layers[i], results);
  }
  return results;
}

void mlp_free(MLP *mlp) {
  for (int i = 0; i < mlp->num_layers; i++) {
    layer_free(mlp->layers[i]);
  }
  free(mlp->layers);
  free(mlp);
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

int main() { test_mlp(); }
