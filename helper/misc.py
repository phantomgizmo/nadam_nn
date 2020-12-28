import numpy as np

def sigmoid(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def make_nabla(delta, activation):
    """Multiplicating delta and activation"""
    result = []
    delta_len = len(delta)
    for i in range(delta_len):
        temp = activation * delta[i]
        result.append(temp)
    return result

def cost_derivative(output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations-y)
    
def cost(output_activations, y):
    return sum(np.square((output_activations-y))) / (2 * len(output_activations))