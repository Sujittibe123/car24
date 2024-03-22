import tensorflow as tf
from tensorflow import keras
import joblib

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255



X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# Using Hidden Layer 

model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)

model.evaluate(X_test_flattened,y_test)

print(model)

# Save the trained model to a file using joblib
joblib.dump(model, 'mnist_trained_model.joblib')

# Load the trained model from the file
# loaded_model = joblib.load('mnist_trained_model.joblib')