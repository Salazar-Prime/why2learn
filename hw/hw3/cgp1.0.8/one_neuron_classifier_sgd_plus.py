"""
Homework 3: Understanding CGP code and implementing SGD + Momentum 
Author: Varun Aggarwal
Last Modified: 5 Feb 2022
"""

import random
import numpy as np
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

    def backprop_and_update_params_one_neuron_model(
        self, y_error_avg, data_tuple_avg, deriv_sigmoid_avg
    ):
        """
        This function is copied over from
        ComputationalGraphPrimer.py Version 1.0.8
        Modified by: Varun Aggarwal

        Modifications:
        Added SGDplusMomentum
        """
        input_vars = self.independent_vars
        vals_for_input_vars_dict = dict(zip(input_vars, list(data_tuple_avg)))
        vals_for_learnable_params = self.vals_for_learnable_params

        # preparing varibles
        step_hist = list(np.zeros(len(self.vals_for_learnable_params)))
        bias_hist = 0

        for i, param in enumerate(self.vals_for_learnable_params):
            ## calculate the next step in the parameter hyperplane

            # representing in same notation as the HW text
            g_tp1 = (
                y_error_avg
                * vals_for_input_vars_dict[input_vars[i]]
                * deriv_sigmoid_avg
            )

            step = self.mu * self.step_hist[i] + self.learning_rate * g_tp1
            self.vals_for_learnable_params[param] += step

            # update step_hist
            self.step_hist[i] = step

        ## Bias momentum step
        self.bias_hist = (
            self.mu * self.bias_hist
            + self.learning_rate * y_error_avg * deriv_sigmoid_avg
        )
        self.bias += self.bias_hist

    def run_training_loop_one_neuron_model(self, training_data):
        """
        This function is copied over from
        ComputationalGraphPrimer.py Version 1.0.8
        Modified by: Varun Aggarwal

        Modifications:
        initializing step_hist and bias_hist
        """
        self.vals_for_learnable_params = {
            param: random.uniform(0, 1) for param in self.learnable_params
        }
        self.bias = random.uniform(0, 1)

        class DataLoader:
            """
            The data loader's job is to construct a batch of randomly chosen samples from the
            training data.  But, obviously, it must first associate the class labels 0 and 1 with
            the training data supplied to the constructor of the DataLoader.   NOTE:  The training
            data is generated in the Examples script by calling 'cgp.gen_training_data()' in the
            ****Utility Functions*** section of this file.  That function returns two normally
            distributed set of number with different means and variances.  One is for key value '0'
            and the other for the key value '1'.  The constructor of the DataLoader associated a'
            class label with each sample separately.
            """

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
            y_preds, deriv_sigmoids = self.forward_prop_one_neuron_model(data_tuples)
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
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(
                map(
                    operator.truediv,
                    data_tuple_avg,
                    [float(len(class_labels))] * len(class_labels),
                )
            )
            self.backprop_and_update_params_one_neuron_model(
                y_error_avg, data_tuple_avg, deriv_sigmoid_avg
            )

        return loss_running_record


# SGD with momentum
cgp = cgpSuperCharged(
    one_neuron_model=True,
    expressions=["xw=ab*xa+bc*xb+cd*xc+ac*xd"],
    output_vars=["xw"],
    dataset_size=5000,
    learning_rate=1e-3,
    training_iterations=40000,
    batch_size=16,
    display_loss_how_often=100,
    debug=True,
    mu=0.9,
)

# Vanilla SGD
cgp_original = ComputationalGraphPrimer(
    one_neuron_model=True,
    expressions=["xw=ab*xa+bc*xb+cd*xc+ac*xd"],
    output_vars=["xw"],
    dataset_size=5000,
    learning_rate=1e-3,
    training_iterations=40000,
    batch_size=16,
    display_loss_how_often=100,
    debug=True,
)

plt.show(block=True)
# Loss with SGDmomentum
cgp.parse_expressions()
training_data = cgp.gen_training_data()
loss_running_record_mu = cgp.run_training_loop_one_neuron_model(training_data)

# Loss with VanillaSGD
cgp_original.parse_expressions()
training_data = cgp_original.gen_training_data()
loss_running_record = cgp_original.run_training_loop_one_neuron_model(training_data)

# Plotting Loss
plt.figure()
plt.plot(loss_running_record_mu, color="red")
plt.plot(loss_running_record)
plt.legend(["SGD plus momentum", "SGD Vanilla"])
plt.title("One Neuron Training")
plt.xlabel("Iterations (Sampled)")
plt.ylabel("Loss")
plt.savefig("../output/one_with_momentum.png")
