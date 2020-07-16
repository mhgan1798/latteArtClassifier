# %% CNN Model Tester
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier")

# %% Test the model on a completely random picture that you chose
model = tf.keras.models.load_model("./models/classifier_v1")


# %% Load Data
test_dir = r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier\data\holdoutDir"


# %% Data pre-processing
IMG_HEIGHT = 150
IMG_WIDTH = 150
# num_images = len(os.listdir(test_dir + "/pretty")) + len(os.listdir(test_dir + "/ugly"))

num_images = len(os.listdir(test_dir + "/unlabelled"))

# Generator for our image
test_image_generator = ImageDataGenerator(rescale=1.0 / 255,)

test_data_gen = test_image_generator.flow_from_directory(
    batch_size=num_images,
    directory=test_dir,
    shuffle=False,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)

# %% Visualise the test image
test_image, _ = next(test_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


plotImages(test_image[0:2])


# %% Make predictions
# ytrue = test_data_gen[0][1].astype("int32")
ytrue = np.repeat("", num_images)
ypred = np.array([i[0] for i in (model.predict(test_data_gen))])
ypred = ypred >= 0.5
ypred = ypred.astype("int32")

ypred, ytrue

# %% Visualise predictions
def plotImages_labels(images_arr, ypreds, yactual):
    labels_ref = ["pretty", "ugly"]

    nimages = images_arr.shape[0]
    ncols = 3
    nrows = np.ceil(nimages / 3).astype("int")

    fig, axes = plt.subplots(ncols=3, nrows=nrows, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for img, ax in zip(images_arr, axes):
        label_idx = np.where((img == images_arr)[:, :, :, 1][:, 0, 0])[0][0]
        ax.annotate(
            s="Predicted: " + str(labels_ref[ypreds[label_idx]]),
            xy=[75, 125],
            color="red",
            size=26,
            ha="center",
        )

        # if ytrue.dtype == "<U1":
        #     actualLabel = "N/A"
        # else:
        #     actualLabel = str(labels_ref[ytrue[label_idx]])

        # ax.annotate(
        #     s="Actual: " + actualLabel, xy=[75, 100], size=18, ha="center",
        # )

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


plotImages_labels(images_arr=test_image[0:30], ypreds=ypred, yactual=ytrue)


# %% Make grade bins for each prediction
preds_array = np.array([i[0] for i in model.predict(test_data_gen)])
grades = np.digitize(preds_array, bins=[-5, -2, 0, 2, 5]) - 1

# %% Visualise grade predictions
def plotImages_grades(images_arr, grades):
    grades_ref = ["A", "B", "C", "D"]

    nimages = images_arr.shape[0]
    ncols = 3
    nrows = np.ceil(nimages / 3).astype("int")

    fig, axes = plt.subplots(ncols=3, nrows=nrows, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for img, ax in zip(images_arr, axes):
        label_idx = np.where((img == images_arr)[:, :, :, 1][:, 0, 0])[0][0]
        ax.annotate(
            s="Grade: " + str(grades_ref[grades[label_idx]]),
            xy=[75, 125],
            color="red",
            size=26,
            ha="center",
        )

        # if ytrue.dtype == "<U1":
        #     actualLabel = "N/A"
        # else:
        #     actualLabel = str(labels_ref[ytrue[label_idx]])

        # ax.annotate(
        #     s="Actual: " + actualLabel, xy=[75, 100], size=18, ha="center",
        # )

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


plotImages_grades(images_arr=test_image[0:30], grades=grades)
