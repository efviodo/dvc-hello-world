import os
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


class ConvNN:

    def __init__(self, train_data_path: str, val_data_path: str, model_dir: str):
        """

        :param train_data_path: path to training data file
        :param val_data_path: path to validation data file
        :param model_dir: directory where model will be saved
        """

        self.model = None
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.model_dir = model_dir
        self.history = None

    @staticmethod
    def prepare_data_array(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads data from CSV and reshape into numpy array
        :param data_path:
        :return:
        """

        # Read data from csv
        df = pd.read_csv(data_path, dtype=np.float32)

        # Reshapes
        x = df.iloc[:, 1:].values.astype('float32')
        x = x.reshape((x.shape[0], 28, 28))
        x = x / 255.0  # Normalize
        x = np.expand_dims(x, -1)  # make sure image have shape (28, 28, 1)
        y = df.iloc[:, 0].values.astype('int32')

        return x, y

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        img.load()
        data = np.asarray(img, dtype="float64")
        data = data / 255.0  # Normalize
        return data

    def build_model(self) -> None:
        """
        Build a simple ConvNN to classify MNIST digits
        :return:
        """

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Print the model summary
        print(model.summary())

        self.model = model

    def train(self) -> None:
        """
        Train model
        :return:
        """

        # 1. build
        self.build_model()

        # 2. compile
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 3. train
        X_train, Y_train = self.prepare_data_array(self.train_data_path)
        X_val, Y_val = self.prepare_data_array(self.val_data_path)
        self.history = self.model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), epochs=10, verbose=1)

    def predict(self, image_path: str) -> int:
        """
        Predict the category of an image using the trained model
        :param image_path: Path to image to predict label
        :return:
        """

        img_np = self.load_image(image_path)
        predictions = self.model.predict(np.expand_dims(img_np, 0))
        return int(np.argmax(predictions[0]))

    def save(self) -> None:
        """
        Save the mmodel
        """

        self.model.save(os.path.join(self.model_dir, 'mnist_model.h5'))
