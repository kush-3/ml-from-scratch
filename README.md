# 🧠 Neural Network from Scratch

A fully functional neural network built using **only NumPy** - no TensorFlow, no PyTorch, no ML frameworks.

## 🎯 Results

**97.58% accuracy on MNIST** with a simple 3-layer network.

## 🏗️ Architecture
```
Input (784) → Hidden (512) → Hidden (512) → Output (10)
```

- **Activation:** ReLU (hidden layers), Softmax (output)
- **Loss:** Cross-Entropy
- **Optimization:** Mini-batch Gradient Descent
- **Weight Init:** He Initialization

## 🔧 What's Implemented from Scratch

| Component | Description |
|-----------|-------------|
| Forward Pass | Matrix multiplications + activations |
| Backpropagation | Gradient computation through all layers |
| ReLU & Softmax | Activation functions |
| Cross-Entropy Loss | Loss calculation |
| Mini-batch SGD | Weight updates with batching |
| He Initialization | Proper weight initialization for ReLU |

## 🚀 Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Train on MNIST
python train_mnist.py
```

## 📊 Training Output
```
Loading MNIST...
Training samples: 60000
Test samples: 10000

Training...
Epoch 1/5, Loss: 0.4523
Epoch 2/5, Loss: 0.2134
Epoch 3/5, Loss: 0.1567
Epoch 4/5, Loss: 0.1198
Epoch 5/5, Loss: 0.0934

Train Accuracy: 0.9812
Test Accuracy: 0.9758
```

## 📚 What I Learned

- How forward propagation actually works (matrix math)
- Backpropagation and chain rule in practice
- Why He initialization matters for ReLU
- How mini-batch gradient descent improves training
- The math behind cross-entropy loss

## 📝 License

MIT

---

**Built by [Kush Patel](https://github.com/kush-3)** as part of learning ML fundamentals from scratch.