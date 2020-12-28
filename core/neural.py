import numpy as np
import random
import helper.misc as misc_func

class Network(object):

    def __init__(self, sizes, learning_rate, beta_1, beta_2):
        np.random.seed(0)
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def reset(self):
        np.random.seed(0)
        self.biases = [np.random.randn(y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def NAdam(self, training_data, epochs, min_error, test_data=None):
        """Return list of test accuracy and output pairs(pair of predicted val and real val) on each epoch.
        training_data contains a tuple of X_train and y_train. (X_train, y_train)
        epochs is the number of maximum epochs for training.
        min_error is minimum error condition for training to stop.
        test_data contains a typle of X_test and y_test. (X_test, y_test)"""
        X_test, y_test = test_data
        X_train, y_train = training_data
        if test_data: n_test = len(X_test)
        n = len(X_train)

        test_acc = [] #accuracy per epoch
        output_pairs = [] #results per epoch

        w_m_prev = [np.zeros(w.shape) for w in self.weights]
        w_v_prev = [np.zeros(w.shape) for w in self.weights]

        b_m_prev = [np.zeros(b.shape) for b in self.biases]
        b_v_prev = [np.zeros(b.shape) for b in self.biases]

        error = 0

        counter = 0

        for j in range(epochs):
            counter = j +1
            print(counter)
            w_m_prev, w_v_prev, b_m_prev, b_v_prev, error = self.update_nadam(training_data, w_m_prev, w_v_prev, b_m_prev, b_v_prev, j + 1)

            if test_data:
                temp_test_acc, results = self.evaluate(test_data)
                test_acc.append(temp_test_acc)
                output_pairs.append(results)

            if error <= min_error: break

        print("Berhenti pada epoch ke : {0}".format(str(counter)))
        print("MSE terakhir bernilai : {0}".format(str(error)))

        return (test_acc, output_pairs)

    def update_nadam(self, batch, w_m_prev, w_v_prev, b_m_prev, b_v_prev, iteration):
        batch_len = len(batch[0])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        new_biases = []
        new_weights = []
        w_m_curr = []
        w_v_curr = []
        b_m_curr = []
        b_v_curr = []
        error = 0

        #calculate nabla_b and nabla_w
        for x, y in zip(batch[0], batch[1]):
            zs, activations = self.feedforward(x)
            delta_nabla_b, delta_nabla_w = self.backprop(zs, activations, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            error += misc_func.cost(activations[-1], y) / batch_len

        #calculate new weights
        for w, nw, w_m, w_v in zip(self.weights, nabla_w, w_m_prev, w_v_prev):
            new_w, temp_w_m_curr, temp_w_v_curr = self.do_nadam(w, nw, w_m, w_v, batch_len, iteration)

            new_weights.append(new_w)
            w_m_curr.append(temp_w_m_curr)
            w_v_curr.append(temp_w_v_curr)

        #calculate new biases
        for b, nb, b_m, b_v in zip(self.biases, nabla_b, b_m_prev, b_v_prev):
            new_b, temp_b_m_curr, temp_b_v_curr = self.do_nadam(b, nb, b_m, b_v, batch_len, iteration)

            new_biases.append(new_b)
            b_m_curr.append(temp_b_m_curr)
            b_v_curr.append(temp_b_v_curr)

        #update weights and biases
        self.weights = new_weights
        self.biases = new_biases

        return (w_m_curr, w_v_curr, b_m_curr, b_v_curr, error)

    def do_nadam(self, param, nabla_param, param_m_prev, param_v_prev, batch_len, iteration):
        """Return the new parameter, param_m, and param_v."""
        temp_param_m_curr = self.beta_1 * param_m_prev + (1 - self.beta_1) * np.array(nabla_param)
        temp_param_v_curr = self.beta_2 * param_v_prev + (1 - self.beta_2) * np.square(np.array(nabla_param))

        param_m_hat = temp_param_m_curr / (1 - np.power(self.beta_1, iteration))
        param_v_hat = temp_param_v_curr / (1 - np.power(self.beta_2, iteration))

        new_param = param - (self.learning_rate / batch_len) * (self.beta_1 * param_m_hat + (((1 - self.beta_1) * np.array(nabla_param)) / (1 - np.power(self.beta_1, iteration)))) / (np.sqrt(param_v_hat) + 10**(-8))

        return new_param, temp_param_m_curr, temp_param_v_curr


    def backprop(self, zs, activations, y):
        """Return nabla of biases and weights of given data.
        Take list of z, activation, 
        and the real outcome as a reference."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # backward pass at the last layer
        delta = misc_func.cost_derivative(activations[-1], y) * \
            misc_func.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = misc_func.make_nabla(delta, activations[-2])

        # backward pass from second to the last layer to the first layer.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = misc_func.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = misc_func.make_nabla(delta, activations[-l-1])

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        X_test, y_test = test_data

        test_results = []

        results = []

        for x, y in zip(X_test, y_test):
            zs, activations = self.feedforward(x)
            nn_out = activations[-1]
            max_index = np.argmax(nn_out)
            pred_val = nn_out[max_index]

            results.append((nn_out, y))
            test_results.append((1 if (pred_val) >= 0.36 else 0, y))

        return (sum(int(x == y) for (x, y) in test_results) / len(y_test), results)

    def feedforward(self, input_data):
        """Return the z and activation of each layer of the network."""
        zs = []
        activation = input_data
        activations = [activation]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = misc_func.sigmoid(z)
            activations.append(activation)
        return zs, activations
    