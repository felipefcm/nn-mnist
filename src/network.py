import numpy as np

# nn = NeuralNetwork([784, 16, 16, 10])


class NeuralNetwork():
    def __init__(self, neurons_per_layer):
        self.num_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer

        a = neurons_per_layer[1:]
        b = neurons_per_layer[:-1]

        self.weights = [
            np.random.randn(current, previous) for current, previous in
            zip(a, b)
        ]

        self.bias = [np.random.randn(y, 1) for y in a]

    def activation_fn(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def cost_derivative(self, output, expected):
        return output - expected

    def activation_derivative(self, x):
        return self.activation_fn(x) * (1 - self.activation_fn(x))

    def feed_forward(self, x):
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, x) + b
            x = self.activation_fn(z)

        return x

    def backprop(self, x, expected):
        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.bias]

        zs = []
        activation = np.array(x)
        activations = [np.array(x)]

        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_fn(z)
            activations.append(activation)

        delta = self.cost_derivative(
            activation, expected) * self.activation_derivative(zs[-1])

        weight_gradients[-1] = np.dot(delta, activations[-2].T)
        bias_gradients[-1] = delta

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            d = self.activation_derivative(z)
            delta = np.dot(self.weights[-layer + 1].T, delta) * d

            weight_gradients[-layer] = np.dot(delta, activations[-layer - 1].T)
            bias_gradients[-layer] = delta

        return (weight_gradients, bias_gradients)

    def adjust(self, lr, weight_gradients, bias_gradients):
        self.weights = [
            w - lr * nw for w, nw in
            zip(self.weights, weight_gradients)
        ]

        self.bias = [
            b - lr * nb for b, nb in
            zip(self.bias, bias_gradients)
        ]
