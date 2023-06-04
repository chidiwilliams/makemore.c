#include "makemore.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

  // bigram_print(bigram);

  // Sample from bigram
  const int num_samples = 10;
  for (int i = 0; i < num_samples; i++) {
    bigram_sample(bigram);
  }

  char *test_words[] = {"andrejq"};
  double num_test_words =
      (double)sizeof(test_words) / (double)sizeof(test_words[0]);
  double average_nll = bigram_average_nll(bigram, test_words, num_test_words);
  // printf("nll/n = %f\n", average_nll);

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

  Value **outputs = mlp_apply(mlp, inputs);

  int num_outputs = layer_outputs[num_layer_outputs - 1];

  for (int i = 0; i < num_outputs; i++) {
    value_print_tree(outputs[i]);
  }

  for (int i = 0; i < num_outputs; i++) {
    value_free(outputs[i]);
  }

  mlp_free(mlp);
}

void print_values(Value **values, int num_values) {
  for (int i = 0; i < num_values; i++) {
    value_print(values[i]);
  }
}

void test_mlp_loss() {
#define NUM_INPUTS 3
#define NUM_LAYERS 2
#define NUM_SAMPLES 4
#define NUM_TRAINING_RUNS 1

  int layer_outputs[NUM_LAYERS] = {1, 1};
  int num_outputs = layer_outputs[NUM_LAYERS - 1];
  MLP *mlp = mlp_init(NUM_INPUTS, layer_outputs, NUM_LAYERS);

  Value *inputs[NUM_SAMPLES][NUM_INPUTS] = {
      {value_init_constant(2), value_init_constant(3), value_init_constant(-1)},
      {value_init_constant(3), value_init_constant(-1),
       value_init_constant(0.5)},
      {value_init_constant(0.5), value_init_constant(1),
       value_init_constant(1)},
      {value_init_constant(1), value_init_constant(1),
       value_init_constant(-1)}};
  Value *outputs[NUM_SAMPLES] = {
      value_init_constant(1), value_init_constant(-1), value_init_constant(-1),
      value_init_constant(1)};

  for (int r = 0; r < NUM_TRAINING_RUNS; r++) {
    Value **predicted_outputs =
        (Value **)allocate(NUM_SAMPLES * sizeof(Value *));
    for (int s = 0; s < NUM_SAMPLES; s++) {
      Value **predicted_outputs_for_input = mlp_apply(mlp, inputs[s]);
      predicted_outputs[s] = predicted_outputs_for_input[0];
      free(predicted_outputs_for_input);
    }

    Value *loss = value_init_constant(0);
    for (int i = 0; i < NUM_SAMPLES; i++) {
      loss = value_add(loss,
                       value_pow(value_minus(predicted_outputs[i], outputs[i]),
                                 value_init_constant(2)));
    }

    value_backward_tree(loss);

    printf("%3d: ", r);
    value_print_tree(loss);

#define LEARNING_RATE -0.001
    for (int i = 0; i < mlp->num_layers; i++) {
      Layer *layer = mlp->layers[i];
      for (int j = 0; j < layer->num_outputs; j++) {
        Neuron *neuron = layer->neurons[j];
        for (int k = 0; k < neuron->num_inputs; k++) {
          neuron->w[k]->data += LEARNING_RATE * neuron->w[k]->grad;
        }
        neuron->b->data += LEARNING_RATE * neuron->b->grad;
      }
    }

    // TODO: free up until the MLP
    value_free(loss);

    // TODO: this should be freed in the loss
    for (int i = 0; i < NUM_SAMPLES; i++) {
      value_free(predicted_outputs[i]);
    }
    free(predicted_outputs);
  }

  // for (int s = 0; s < NUM_SAMPLES; s++) {
  //   Value **predicted_outputs = mlp_apply(mlp, inputs[s]);
  //   // printf("Predicted: \n");
  //   // print_values(predicted_outputs, num_outputs);
  //
  //   for (int i = 0; i < num_outputs; i++) {
  //     value_free(predicted_outputs[i]);
  //   }
  //   free(predicted_outputs);
  // }

  for (int i = 0; i < NUM_SAMPLES; i++) {
    for (int j = 0; j < NUM_INPUTS; j++) {
      value_free(inputs[i][j]);
    }
  }

  for (int i = 0; i < NUM_SAMPLES; i++) {
    value_free(outputs[i]);
  }

  mlp_free(mlp);
}

void neuron_free_weight_result(Value *value, int num_inputs_left) {
  if (num_inputs_left == 0) {
    value_free(value);
    return;
  }

  neuron_free_weight_result(value->left_child, num_inputs_left - 1);
  value_free(value->right_child);
  value_free(value);
}

void neuron_free_result(Neuron *neuron, Value *result) {
  Value *tanh_value = result;
  Value *add_b_value = tanh_value->left_child;
  neuron_free_weight_result(add_b_value->left_child, neuron->num_inputs - 1);
  value_free(add_b_value);
  value_free(tanh_value);
}

void layer_free_result(Layer *layer, Value **outputs) {
  for (int i = 0; i < layer->num_outputs; i++) {
    neuron_free_result(layer->neurons[i], outputs[i]);
  }
  free(outputs);
}

void mlp_neuron_free_result(Neuron *neuron, Value *result, int is_first,
                            MLP *mlp, int layer_index);

void mlp_neuron_free_weight_result(Value *value, int num_inputs_left,
                                   int is_first, MLP *mlp, int layer_index) {
  if (num_inputs_left == 0) {
    if (is_first) {
      Layer *nested_layer = mlp->layers[layer_index - 1];
      Neuron *neuron = nested_layer->neurons[num_inputs_left];

      if (layer_index == 1) {
        neuron_free_result(neuron, value->right_child);
      } else {
        mlp_neuron_free_result(neuron, value->right_child, is_first, mlp,
                               layer_index - 1);
      }
    }
    value_free(value);
    return;
  }

  mlp_neuron_free_weight_result(value->left_child, num_inputs_left - 1,
                                is_first, mlp, layer_index);
  if (is_first) {
    Layer *nested_layer = mlp->layers[layer_index - 1];
    Neuron *neuron = nested_layer->neurons[num_inputs_left];
    neuron_free_result(neuron, value->right_child->right_child);
  }
  value_free(value->right_child);
  value_free(value);
}

void mlp_neuron_free_result(Neuron *neuron, Value *result, int is_first,
                            MLP *mlp, int layer_index) {
  Value *tanh_value = result;
  Value *add_b_value = tanh_value->left_child;
  mlp_neuron_free_weight_result(add_b_value->left_child, neuron->num_inputs - 1,
                                is_first, mlp, layer_index);
  value_free(add_b_value);
  value_free(tanh_value);
}

void mlp_layer_free_result(Layer *layer, Value **outputs, int layer_index,
                           MLP *mlp) {
  for (int i = 0; i < layer->num_outputs; i++) {
    mlp_neuron_free_result(layer->neurons[i], outputs[i], i == 0, mlp,
                           layer_index);
  }
  free(outputs);
}

void mlp_free_outputs(MLP *mlp, Value **outputs) {
  Layer *output_layer = mlp->layers[mlp->num_layers - 1];
  mlp_layer_free_result(output_layer, outputs, mlp->num_layers - 1, mlp);
}

void free_loss_sum() {}

void free_loss(Value *loss, int depth) {
  if (depth == 0) {
    return;
  }

  free_loss(loss->left_child, depth - 1);

  Value *pow_value = loss->right_child;
  Value *sum_value = pow_value->left_child;
  Value *exponent_value = pow_value->right_child;
  Value *negate_value = sum_value->right_child;
  Value *minus_one_value = negate_value->right_child;

  value_print(minus_one_value);
  free(minus_one_value);

  value_print(negate_value);
  free(negate_value);

  value_print(sum_value);
  free(sum_value);

  value_print(exponent_value);
  free(exponent_value);

  value_print(pow_value);
  free(pow_value);

  value_print(loss);
  free(loss);
}

int main(int argc, char *argv[]) {
  srand(0);

  char *type = NULL;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--type") == 0) {
      type = argv[i + 1];
      i++;
    }
  }

  if (type != NULL && (strcmp(type, "bigram") == 0)) {
    test_bigram();
  } else {
#define num_layer_outputs 3
#define num_inputs 3
#define num_samples 1
    int layer_outputs[num_layer_outputs] = {1, 1, 1};
    Value *inputs[num_samples][num_inputs] = {
        {value_init_constant(1), value_init_constant(2),
         value_init_constant(3)},
        // {value_init_constant(3), value_init_constant(2),
        //  value_init_constant(1)},
        // {value_init_constant(3), value_init_constant(2),
        //  value_init_constant(1)},
    };
    Value *outputs[num_samples] = {
        value_init_constant(1),
        // value_init_constant(-1),
        // value_init_constant(2),
    };

    MLP *mlp = mlp_init(num_inputs, layer_outputs, num_layer_outputs);

    Value ***sample_outputs =
        (Value ***)allocate(num_samples * sizeof(Value **));
    for (int i = 0; i < num_samples; i++) {
      Value **outputs = mlp_apply(mlp, inputs[i]);
      sample_outputs[i] = outputs;
    }

    Value *loss = value_init_constant(0);
    for (int i = 0; i < num_samples; i++) {
      loss = value_add(loss,
                       value_pow(value_minus(sample_outputs[i][0], outputs[i]),
                                 value_init_constant(2)));
    }

    value_print_tree(loss);

    free_loss(loss, num_samples);

    for (int i = 0; i < num_samples; i++) {
      mlp_free_outputs(mlp, sample_outputs[i]);
    }
    free(sample_outputs);

    for (int i = 0; i < num_samples; i++) {
      value_free(outputs[i]);
    }

    for (int i = 0; i < num_samples; i++) {
      for (int j = 0; j < num_inputs; j++) {
        value_free(inputs[i][j]);
      }
    }

    mlp_free(mlp);
  }

  return 0;
}
