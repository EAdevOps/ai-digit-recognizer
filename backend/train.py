import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1) Get the homework (MNIST digit pictures)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2) Clean it so the brain understands (turn 0-255 gray into 0-1 and add a channel)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = x_train[..., None]  # (28,28) -> (28,28,1)
x_test  = x_test[..., None]

# 3) Build the brain: layers that look for tiny shapes, then bigger shapes
model = models.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 numbers = 0..9
])

# 4) Tell the brain how to learn (optimizer, what’s right/wrong)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 5) Practice for 3 rounds (epochs)
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=2)

# 6) Test: how good are you?
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# 7) Save your memory so we can use it later
os.makedirs("model", exist_ok=True)
model.save("model/mnist_cnn.keras")
print("Saved model to model/mnist_cnn.keras")
