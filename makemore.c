#include "makemore.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void *allocate(size_t size) {
  void *result = malloc(size);
  if (result == NULL) {
    exit(1);
  }
  return result;
}

// 26 + "."
#define ALPHABET_SIZE 27
#define CHAR_TO_INDEX(char) (char - 'a' + 1)
#define INDEX_TO_CHAR(index) ('a' + index - 1)

double **bigram_init() {
  double **bigram = (double **)allocate(ALPHABET_SIZE * sizeof(double *));
  for (int i = 0; i < ALPHABET_SIZE; i++) {
    bigram[i] = (double *)allocate(ALPHABET_SIZE * sizeof(double));
    for (int j = 0; j < ALPHABET_SIZE; j++) {
      bigram[i][j] = 0;
    }
  }
  return bigram;
}

void bigram_add_word(double **bigram, char *word, int num_chars) {
  // start token and first character
  bigram[0][CHAR_TO_INDEX(word[0])] += 1;

  // middle characters
  for (int i = 1; i < num_chars; i++) {
    bigram[CHAR_TO_INDEX(word[i - 1])][CHAR_TO_INDEX(word[i])] += 1;
  }

  // last character and end token
  bigram[CHAR_TO_INDEX(word[num_chars - 1])][0] += 1;
}

void bigram_normalize(double **bigram) {
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
}

void bigram_print(double **bigram) {
  for (int i = 0; i < ALPHABET_SIZE; i++) {
    for (int j = 0; j < ALPHABET_SIZE; j++) {
      printf("%.5f ", bigram[i][j]);
    }
    printf("\n");
  }
}

void bigram_free(double **bigram) {
  for (int i = 0; i < ALPHABET_SIZE; i++) {
    free(bigram[i]);
  }
  free(bigram);
}

static int sample_multinomial(double *values, int size);

void bigram_sample(double **bigram) {
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

double bigram_average_nll(double **bigram, char **words, int num_words) {
  double log_likelihood = 0;
  double n = 0;

  for (int i = 0; i < num_words; i++) {
    char *word = words[i];

    // start token and first character
    log_likelihood += log(bigram[0][CHAR_TO_INDEX(word[0])]);

    // middle characters
    int i;
    for (i = 1; word[i] != '\0'; i++) {
      log_likelihood +=
          log(bigram[CHAR_TO_INDEX(word[i - 1])][CHAR_TO_INDEX(word[i])]);
    }

    // last character and end token
    log_likelihood += log(bigram[CHAR_TO_INDEX(word[i - 2])][0]);

    // i + 1 is the number of sequences in word with length i
    n += i + 1;
  }
  double nll = -log_likelihood;
  return nll / n;
}

static int sample_multinomial(double *values, int size) {
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

static Value *value_init(double data, enum ValueType type) {
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

static Value *value_init_binary(double data, enum ValueType type,
                                Value *leftChild, Value *rightChild) {
  Value *value = value_init(data, type);
  value->left_child = leftChild;
  value->right_child = rightChild;
  return value;
}

static Value *value_init_unary(double data, enum ValueType type, Value *child) {
  Value *value = value_init(data, type);
  value->left_child = child;
  return value;
}

Value *value_add(Value *value1, Value *value2) {
  return value_init_binary(value1->data + value2->data, ADD, value1, value2);
}

Value *value_times(Value *value1, Value *value2) {
  return value_init_binary(value1->data * value2->data, MULTIPLY, value1,
                           value2);
}

Value *value_negate(Value *value) {
  return value_times(value, value_init_constant(-1));
}

Value *value_minus(Value *value1, Value *value2) {
  return value_add(value1, value_negate(value2));
}

Value *value_tanh(Value *value) {
  return value_init_unary(tanh(value->data), TANH, value);
}

Value *value_pow(Value *value, Value *power) {
  return value_init_binary(pow(value->data, power->data), POW, value, power);
}

static void value_backward(Value *value) {
  switch (value->type) {
  case CONSTANT: {
    break;
  }
  case ADD: {
    value->left_child->grad += 1.0 * value->grad;
    value->right_child->grad += 1.0 * value->grad;
    break;
  }
  case MULTIPLY: {
    value->left_child->grad += value->right_child->data * value->grad;
    value->right_child->grad += value->left_child->data * value->grad;
    break;
  }
  case TANH: {
    double d = value->left_child->data;
    double t = (exp(2 * d) - 1) / (exp(2 * d) + 1);
    value->left_child->grad += (1 - pow(t, 2)) * value->grad;
    break;
  }
  case POW: {
    value->left_child->grad +=
        value->right_child->data *
        pow(value->left_child->data, value->right_child->data - 1) *
        value->grad;
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
  case POW:
    label = "pow";
  }

  printf("[%4s | %.10f | %f]\n", label, value->data, value->grad);
}

static void value_print_tree_at_depth(Value *value, int depth) {
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

typedef struct ValueListNode {
  Value *value;
  struct ValueListNode *next;
} ValueListNode;

static ValueListNode *valueitem_init(Value *value) {
  ValueListNode *item = (ValueListNode *)allocate(sizeof(ValueListNode));
  item->value = value;
  item->next = NULL;
  return item;
}

static void sort_topological(ValueListNode **sorted, ValueListNode **visited,
                             Value *value) {
  if (value == NULL) {
    return;
  }

  if (*visited != NULL) {
    ValueListNode *current = *visited;
    while (current->next != NULL) {
      if (current->value == value) {
        return;
      }
      current = current->next;
    }
  }

  ValueListNode *visitedItem = valueitem_init(value);
  if (*visited == NULL) {
    *visited = visitedItem;
  } else {
    ValueListNode *current = *visited;
    while (current->next != NULL) {
      current = current->next;
    }
    current->next = visitedItem;
  }

  sort_topological(sorted, visited, value->left_child);
  sort_topological(sorted, visited, value->right_child);

  ValueListNode *sortedItem = valueitem_init(value);
  if (*sorted == NULL) {
    *sorted = sortedItem;
  } else {
    sortedItem->next = *sorted;
    *sorted = sortedItem;
  }
}

static void valueitem_free(ValueListNode *valueItem) {
  if (valueItem == NULL) {
    return;
  }
  valueitem_free(valueItem->next);
  free(valueItem);
}

void value_backward_tree(Value *value) {
  ValueListNode *sorted = NULL;
  ValueListNode *visited = NULL;

  sort_topological(&sorted, &visited, value);

  value->grad = 1;

  for (ValueListNode *current = sorted; current != NULL;
       current = current->next) {
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

static double random_weight() {
  return ((double)random() * 2 / (double)RAND_MAX) - 1;
}

Neuron *neuron_init(int num_inputs) {
  Neuron *neuron = (Neuron *)allocate(sizeof(Neuron));
  neuron->num_inputs = num_inputs;
  neuron->b = value_init_constant_with_label(random_weight(), "b");

  Value **w = (Value **)allocate(num_inputs * sizeof(Value));
  for (int i = 0; i < num_inputs; i++) {
    w[i] = value_init_constant_with_label(random_weight(), "w");
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
  Value **outputs = inputs;
  for (int i = 0; i < mlp->num_layers; i++) {
    Value **next_results = layer_apply(mlp->layers[i], outputs);
    // If results was from a previous layer, free the array (but not the
    // contents)
    if (i > 0) {
      free(outputs);
    }
    outputs = next_results;
  }
  return outputs;
}

void mlp_free(MLP *mlp) {
  for (int i = 0; i < mlp->num_layers; i++) {
    layer_free(mlp->layers[i]);
  }
  free(mlp->layers);
  free(mlp);
}
