# %% CNN Model Tester
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt

import tempfile
from shutil import copyfile

os.chdir(r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier")

# %% Load in the pre-trained CNN model
model = tf.keras.models.load_model("./models/classifier_v1")


# %% Get a random image from the pool of all images
# list all images from ugly dir
copyFromDir = r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier\data\validationDir\ugly"
files = np.array([str(copyFromDir + "/" + i) for i in os.listdir(copyFromDir)])

# list all images from pretty dir
copyFromDir = r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier\data\validationDir\pretty"
files2 = np.array([str(copyFromDir + "/" + i) for i in os.listdir(copyFromDir)])

# Combine all paths and shuffle
filesList = np.concatenate([files.tolist(), files2.tolist()])
np.random.shuffle(filesList)

# Choose a random image based on the previous shuffling
filename = filesList[0]


# %% Load the single selected image into a temporary directory
temp_dir = tempfile.TemporaryDirectory()
os.mkdir(temp_dir.name + "/unlabelled")
copyfile(filename, str(temp_dir.name + "/unlabelled/img.png"))

# %% Create a generator object from the image in the temp dir
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Generator for our image
temp_image_generator = ImageDataGenerator(rescale=1.0 / 255,)

temp_data_gen = temp_image_generator.flow_from_directory(
    batch_size=1,
    directory=temp_dir.name,
    shuffle=False,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)

temp_image, _ = next(temp_data_gen)

# %% Assign the prediction into a grade bin
preds_array = np.array([i[0] for i in model.predict(temp_data_gen)])
grade = np.digitize(preds_array, bins=[-10, -2, 0, 1.5, 3, 4, 5, 10]) - 1


# %% Visualise grade predictions
def assignImgGrade(img, grade):
    grades_ref = ["A+", "A", "B", "C", "D", "E", "F"]

    fig, ax = plt.subplots()

    ax.annotate(
        s="GRADE: " + str(grades_ref[grade[0]]),
        xy=[75, 140],
        bbox={
            "facecolor": "#303030",
            "alpha": 0.9,
            "pad": 0.3,
            "boxstyle": "round, pad=0.3",
            "lw": 0,
        },
        color="#fffae0",
        fontsize=24,
        fontname="Helvetica",
        fontweight="bold",
        ha="center",
    )

    ax.imshow(img)
    ax.axis("off")

    plt.tight_layout()
    plt.show()


assignImgGrade(img=temp_image[0], grade=grade)

# %% Clean up the temporary directory
temp_dir.cleanup()
