"""
Homework 3: Understanding CGP code and implementing SGD + Momentum 
Author: Varun Aggarwal
Last Modified: 5 Feb 2022
"""

import random
import numpy as np
import sys
import operator
import matplotlib.pyplot as plt
from ComputationalGraphPrimer import *

seed = 0
random.seed(seed)
np.random.seed(seed)

# inherited class
class cgpSuperCharged(ComputationalGraphPrimer):
    def __init__(self, mu=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def backprop_and_update_params_multi_neuron_model(self, y_error, class_labels):
        """
        This function is copied over from
        ComputationalGraphPrimer.py Version 1.0.8
        Modified by: Varun Aggarwal

        Modifications:
        Added SGDplusMomentum
        """
        # backproped prediction error:
        pred_err_backproped_at_layers = {i: [] for i in range(1, self.num_layers - 1)}
        pred_err_backproped_at_layers[self.num_layers - 1] = [y_error]

        for back_layer_index in reversed(range(1, self.num_layers)):
            input_vals = self.forw_prop_vals_at_layers[back_layer_index - 1]
            input_vals_avg = [sum(x) for x in zip(*input_vals)]
            input_vals_avg = list(
                map(
                    operator.truediv,
                    input_vals_avg,
                    [float(len(class_labels))] * len(class_labels),
                )
            )
            deriv_sigmoid = self.gradient_vals_for_layers[back_layer_index]
            deriv_sigmoid_avg = [sum(x) for x in zip(*deriv_sigmoid)]
            deriv_sigmoid_avg = list(
                map(
                    operator.truediv,
                    deriv_sigmoid_avg,
                    [float(len(class_labels))] * len(class_labels),
                )
            )
            vars_in_layer = self.layer_vars[back_layer_index]  ## a list like ['xo']
            vars_in_next_layer_back = self.layer_vars[
                back_layer_index - 1
            ]  ## a list like ['xw', 'xz']

            layer_params = self.layer_params[back_layer_index]
            ## note that layer_params are stored in a dict like
            ##     {1: [['ap', 'aq', 'ar', 'as'], ['bp', 'bq', 'br', 'bs']], 2: [['cp', 'cq']]}
            ## "layer_params[idx]" is a list of lists for the link weights in layer whose output nodes are in layer "idx"
            transposed_layer_params = list(
                zip(*layer_params)
            )  ## creating a transpose of the link matrix

            backproped_error = [None] * len(vars_in_next_layer_back)
            for k, varr in enumerate(vars_in_next_layer_back):
                for j, var2 in enumerate(vars_in_layer):
                    backproped_error[k] = sum(
                        [
                            self.vals_for_learnable_params[
                                transposed_layer_params[k][i]
                            ]
                            * pred_err_backproped_at_layers[back_layer_index][i]
                            for i in range(len(vars_in_layer))
                        ]
                    )
            pred_err_backproped_at_layers[back_layer_index - 1] = backproped_error
            input_vars_to_layer = self.layer_vars[back_layer_index - 1]
            for j, var in enumerate(vars_in_layer):
                layer_params = self.layer_params[back_layer_index][j]
                for i, param in enumerate(layer_params):

                    # representing in same notation as the HW text
                    g_tp1 = (
                        input_vals_avg[i]
                        * pred_err_backproped_at_layers[back_layer_index][j]
                    ) * deriv_sigmoid_avg[j]

                    step = self.mu * self.step_hist[i] + self.learning_rate * g_tp1
                    self.vals_for_learnable_params[param] += step

                    # update step_hist
                    self.step_hist[i] = step

            ## Bias momentum step
            self.bias_hist = self.mu * self.bias_hist + self.learning_rate * sum(
                pred_err_backproped_at_layers[back_layer_index]
            ) * sum(deriv_sigmoid_avg) / len(deriv_sigmoid_avg)
            self.bias[back_layer_index - 1] += self.bias_hist

    def run_training_loop_multi_neuron_model(self, training_data):
        """
        This function is copied over from
        ComputationalGraphPrimer.py Version 1.0.8
        Modified by: Varun Aggarwal

        Modifications:
        initializing step_hist and bias_hist
        """

        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):
                cointoss = random.choice([0, 1])
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)

            def getbatch(self):
                batch_data, batch_labels = [], []
                maxval = 0.0
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval:
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item / maxval for item in batch_data]
                batch = [batch_data, batch_labels]
                return batch

        ## We must initialize the learnable parameters
        self.vals_for_learnable_params = {
            param: random.uniform(0, 1) for param in self.learnable_params
        }
        self.bias = [random.uniform(0, 1) for _ in range(self.num_layers - 1)]

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_literations = 0.0

        # preparing varibles
        self.bias_hist = 0
        self.step_hist = list(np.zeros(len(self.learnable_params)))

        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[
                self.num_layers - 1
            ]
            y_preds = [
                item for sublist in predicted_labels_for_batch for item in sublist
            ]
            loss = sum(
                [
                    (abs(class_labels[i] - y_preds[i])) ** 2
                    for i in range(len(class_labels))
                ]
            )
            loss_avg = loss / float(len(class_labels))
            avg_loss_over_literations += loss_avg
            if i % (self.display_loss_how_often) == 0:
                avg_loss_over_literations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_literations)
                print("[iter=%d]  loss = %.4f" % (i + 1, avg_loss_over_literations))
                avg_loss_over_literations = 0.0
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            self.backprop_and_update_params_multi_neuron_model(
                y_error_avg, class_labels
            )
        return loss_running_record


# SGD with momentum
cgp = cgpSuperCharged(
    num_layers=3,
    layers_config=[4, 2, 1],
    expressions=[
        "xw=ap*xp+aq*xq+ar*xr+as*xs",
        "xz=bp*xp+bq*xq+br*xr+bs*xs",
        "xo=cp*xw+cq*xz",
    ],
    output_vars=["xo"],
    dataset_size=5000,
    learning_rate=1e-3,
    training_iterations=40000,
    batch_size=8,
    display_loss_how_often=100,
    debug=True,
    mu=0.9,
)

# Vanilla SGD
cgp_original = ComputationalGraphPrimer(
    num_layers=3,
    layers_config=[4, 2, 1],
    expressions=[
        "xw=ap*xp+aq*xq+ar*xr+as*xs",
        "xz=bp*xp+bq*xq+br*xr+bs*xs",
        "xo=cp*xw+cq*xz",
    ],
    output_vars=["xo"],
    dataset_size=5000,
    learning_rate=1e-3,
    training_iterations=40000,
    batch_size=8,
    display_loss_how_often=100,
    debug=True,
)

# Loss with SGDmomentum
cgp.parse_multi_layer_expressions()
training_data = cgp.gen_training_data()
loss_running_record_mu = cgp.run_training_loop_multi_neuron_model(training_data)

# Loss with VanillaSGD
cgp_original.parse_multi_layer_expressions()
training_data = cgp_original.gen_training_data()
loss_running_record = cgp_original.run_training_loop_multi_neuron_model(training_data)

# Plotting Loss
plt.figure()
plt.plot(loss_running_record_mu, color="red")
plt.plot(loss_running_record)
plt.legend(["SGD plus momentum", "SGD Vanilla"])
plt.title("Multi Neuron Training")
plt.xlabel("Iterations (Sampled)")
plt.ylabel("Loss")
plt.savefig("../output/multi_with_momentum.png")
# plt.show()
