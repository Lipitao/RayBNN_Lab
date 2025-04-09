import numpy as np

# Load predicted labels from 1.csv
predicted_labels = np.loadtxt("./mnist/1.csv", delimiter=",", dtype=float)

# Convert probabilities to predicted labels (index of the highest probability)
predicted_labels = np.argmax(predicted_labels, axis=1)

# Load true labels from mnist_test_y.dat
true_labels = np.loadtxt("../test_data/mnist/mnist_test_y.dat", delimiter=",", dtype=int)
true_labels = np.argmax(true_labels, axis=1)


# Ensure the lengths match
if len(predicted_labels) != len(true_labels):
    raise ValueError("The number of predicted labels and true labels do not match!")

# Calculate accuracy
accuracy = np.sum(predicted_labels == true_labels) / len(true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")