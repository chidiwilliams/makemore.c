#include <stdlib.h>

void *allocate(size_t size);

double **bigram_init();
void bigram_add_word(double **bigram, char *word, int num_chars);
void bigram_print(double **bigram);
void bigram_normalize(double **bigram);
void bigram_sample(double **bigram);
double bigram_average_nll(double **bigram, char **words, int num_words);
void bigram_free(double **bigram);

enum ValueType { CONSTANT, ADD, MULTIPLY, TANH, POW };

typedef struct Value {
  enum ValueType type;
  double data;
  double grad;
  char *label;
  struct Value *left_child;
  struct Value *right_child;
} Value;

Value *value_init_constant(double data);
Value *value_init_constant_with_label(double data, char *label);
Value *value_add(Value *value1, Value *value2);
Value *value_minus(Value *value1, Value *value2);
Value *value_times(Value *value1, Value *value2);
Value *value_pow(Value *value, Value *power);
Value *value_tanh(Value *value);
void value_backward_tree(Value *value);
void value_print_tree(Value *value);
void value_print(Value *value);
void value_free_tree(Value *value);
void value_free(Value *value);

typedef struct Neuron {
  int num_inputs;
  Value **w;
  Value *b;
} Neuron;

Neuron *neuron_init(int num_inputs);
void neuron_free(Neuron *neuron);
void neuron_print(Neuron *neuron);
Value *neuron_apply(Neuron *neuron, Value **inputs);

typedef struct Layer {
  int num_inputs;
  int num_outputs;
  Neuron **neurons;
} Layer;

Layer *layer_init(int num_inputs, int num_outputs);
void layer_free(Layer *layer);
Value **layer_apply(Layer *layer, Value **inputs);

typedef struct MLP {
  int num_layers;
  Layer **layers;
} MLP;

MLP *mlp_init(int num_inputs, int *layer_outputs, int num_layer_outputs);
Value **mlp_apply(MLP *mlp, Value **inputs);
void mlp_free(MLP *mlp);
