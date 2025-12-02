import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("saved_model/mnist_cnn.keras")

# Load MNIST test set
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., None]

# Pick any example
index = 123  # change this number to test different images
img = x_test[index]
true_label = y_test[index]

# Predict
pred = model.predict(img.reshape(1, 28, 28, 1))
predicted_label = np.argmax(pred)

# Show output
print("True Label:", true_label)
print("Predicted Label:", predicted_label)

plt.imshow(img.reshape(28,28), cmap="gray")
plt.title(f"Predicted: {predicted_label},   Actual: {true_label}")
plt.axis('off')
plt.show()
