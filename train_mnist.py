"""
Train the neural network on MNIST dataset
"""

from neural_network import NeuralNetwork
from torchvision import datasets, transforms
import numpy as np

# Load MNIST
print("Loading MNIST...")
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./mnist_data', train=False, download=True, transform=transform)

# Convert to numpy and normalize
X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
y_train = train_dataset.targets.numpy()

X_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
y_test = test_dataset.targets.numpy()

# One-hot encode labels
Y_train = np.zeros((y_train.shape[0], 10))
Y_train[np.arange(y_train.shape[0]), y_train] = 1

Y_test = np.zeros((y_test.shape[0], 10))
Y_test[np.arange(y_test.shape[0]), y_test] = 1

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Train
nn = NeuralNetwork()
print("\nTraining...")
nn.train(X_train, Y_train, epochs=5, learning_rate=0.1, batch_size=64)

# Evaluate
train_acc = nn.accuracy(X_train, Y_train)
test_acc = nn.accuracy(X_test, Y_test)

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
