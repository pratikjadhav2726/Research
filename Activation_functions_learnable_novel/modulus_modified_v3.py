import numpy as np
import matplotlib.pyplot as plt
# Define the Learnable Modified Modulus v3 Activation Function
class LearnableModifiedModulusV3:
    def __init__(self, init_range=0.2):
        self.threshold = init_range  # Learnable parameter for range

    def forward(self, x):
        return np.where((x <= -self.threshold) | (x >= self.threshold), np.abs(x), 0)

    def update(self, grad, lr=0.01):
        self.threshold -= lr * grad  # Simple gradient update for demonstration
        self.threshold = np.clip(self.threshold, 0.01, 1.0)  # Keep within reasonable bounds

# Train the function approximation neural network with Learnable Modified Modulus v3 activation
def train_network_with_learnable_modified_modulus_v3(X, y, epochs=500, learning_rate=0.01):
    np.random.seed(42)
    input_size = X.shape[1]
    hidden_size1 = 10  # First hidden layer
    hidden_size2 = 10  # Second hidden layer
    output_size = 1

    # Initialize weights, biases, and learnable activation function
    W1 = np.random.randn(input_size, hidden_size1) * 0.1
    b1 = np.zeros((1, hidden_size1))
    W2 = np.random.randn(hidden_size1, hidden_size2) * 0.1
    b2 = np.zeros((1, hidden_size2))
    W3 = np.random.randn(hidden_size2, output_size) * 0.1
    b3 = np.zeros((1, output_size))
    activation = LearnableModifiedModulusV3(init_range=0.2)  # Initial threshold

    loss_history = []

    for epoch in range(epochs):
        # Forward pass
        Z1 = np.dot(X, W1) + b1
        A1 = activation.forward(Z1)

        Z2 = np.dot(A1, W2) + b2
        A2 = activation.forward(Z2)

        Z3 = np.dot(A2, W3) + b3
        y_pred = Z3

        # Compute loss (Mean Squared Error)
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(loss)

        # Backpropagation
        dZ3 = 2 * (y_pred - y) / y.shape[0]
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.dot(dZ3, W3.T) * np.where((Z2 <= -activation.threshold) | (Z2 >= activation.threshold), np.sign(Z2), 0)
        dZ2 = dA2
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, W2.T) * np.where((Z1 <= -activation.threshold) | (Z1 >= activation.threshold), np.sign(Z1), 0)
        dZ1 = dA1
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Compute gradient for learnable parameter and update
        activation_grad = np.mean(dA1)  # Approximate gradient for threshold parameter
        activation.update(activation_grad, lr=learning_rate)

        # Update weights and biases
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

    return loss_history, activation.threshold

# Train the function approximation neural network with Learnable Modified Modulus v3 activation
loss_history_learnable_v3, final_threshold = train_network_with_learnable_modified_modulus_v3(
    X_simple, y_simple, epochs=500, learning_rate=0.01
)

# Plot training loss for learnable modified modulus v3
plt.figure(figsize=(8, 5))
plt.plot(loss_history_learnable_v3, label=f"Learnable Modified Modulus v3 (Final Threshold: {final_threshold:.3f})")

plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss for Learnable Modified Modulus v3 in Curve Fitting")
plt.legend()
plt.grid(True)
plt.show()