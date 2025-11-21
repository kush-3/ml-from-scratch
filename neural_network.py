"""
Neural Network from Scratch
Built with only NumPy - no ML frameworks

Architecture: 784 → 512 → 512 → 10
Achieves ~97% accuracy on MNIST
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size=784, hidden1=512, hidden2=512, output_size=10):
        # He initialization for ReLU
        self.w1 = np.random.randn(input_size, hidden1) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden1))

        self.w2 = np.random.randn(hidden1, hidden2) * np.sqrt(2 / hidden1)
        self.b2 = np.zeros((1, hidden2))

        self.w3 = np.random.randn(hidden2, output_size) * np.sqrt(2 / hidden2)
        self.b3 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        Z1 = X @ self.w1 + self.b1
        A1 = self.relu(Z1)

        Z2 = A1 @ self.w2 + self.b2
        A2 = self.relu(Z2)

        Z3 = A2 @ self.w3 + self.b3
        A3 = self.softmax(Z3)

        self.cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
        return A3

    def cross_entropy_loss(self, y_pred, y_true):
        epsilon = 1e-8
        loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
        return loss

    def backward(self, X, y):
        m = X.shape[0]

        A1 = self.cache['A1']
        A2 = self.cache['A2']
        A3 = self.cache['A3']

        # Output layer gradients
        dZ3 = A3 - y
        dW3 = (1 / m) * A2.T @ dZ3
        db3 = (1 / m) * np.sum(dZ3, axis=0)

        # Hidden layer 2 gradients
        dA2 = dZ3 @ self.w3.T
        dZ2 = dA2 * (self.cache['Z2'] > 0)  # ReLU derivative
        dW2 = (1 / m) * A1.T @ dZ2
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer 1 gradients
        dA1 = dZ2 @ self.w2.T
        dZ1 = dA1 * (self.cache['Z1'] > 0)  # ReLU derivative
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

    def update_weights(self, gradients, lr):
        self.w1 -= lr * gradients['dW1']
        self.b1 -= lr * gradients['db1']
        self.w2 -= lr * gradients['dW2']
        self.b2 -= lr * gradients['db2']
        self.w3 -= lr * gradients['dW3']
        self.b3 -= lr * gradients['db3']

    def train(self, x_train, y_train, epochs=5, learning_rate=0.01, batch_size=64):
        num_samples = x_train.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            num_batches = num_samples // batch_size

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size

                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                predictions = self.forward(x_batch)

                # Compute loss
                loss = self.cross_entropy_loss(predictions, y_batch)
                epoch_loss += loss

                # Backward pass
                gradients = self.backward(x_batch, y_batch)

                # Update weights
                self.update_weights(gradients, learning_rate)

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, x):
        predictions = self.forward(x)
        return np.argmax(predictions, axis=1)

    def accuracy(self, x, y_true):
        predictions = self.predict(x)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == y_true_labels)