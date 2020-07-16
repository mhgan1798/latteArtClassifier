# %% CNN Model Generator
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier")

# %% Load Data
train_dir = r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier\data\trainDir"

validation_dir = r"C:\Users\HowardG\Google Drive\pythonProjects\development\latteArtClassifier\data\validationDir"

# %% Load categories
train_pretty_dir = os.path.join(train_dir, "pretty")
train_ugly_dir = os.path.join(train_dir, "ugly")

validation_pretty_dir = os.path.join(validation_dir, "pretty")
validation_ugly_dir = os.path.join(validation_dir, "ugly")

# %% Understand the data
num_pretty_tr = len(os.listdir(train_pretty_dir))
num_ugly_tr = len(os.listdir(train_ugly_dir))

num_pretty_val = len(os.listdir(validation_pretty_dir))
num_ugly_val = len(os.listdir(validation_ugly_dir))

total_train = num_pretty_tr + num_ugly_tr
total_val = num_pretty_val + num_ugly_val


# %%
print("total training pretty images:", num_pretty_tr)
print("total training ugly images:", num_ugly_tr)

print("total validation pretty images:", num_pretty_val)
print("total validation ugly images:", num_ugly_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# %% Set up variables for model for convenience
batch_size = 20
epochs = 1000
IMG_HEIGHT = 150
IMG_WIDTH = 150

# %% Data pre-processing
# Generator for our training data
train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    channel_shift_range=0.15,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.25,
)

# Generator for our validation data
validation_image_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    channel_shift_range=0.15,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.25,
)

# %% Pre-process using the generators - Training data
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)

# %% Pre-process using the generators - Validation data
val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)

# %% Visualise the training and validation images
sample_training_images, _ = next(train_data_gen)
sample_validation_images, _ = next(val_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(4, 5, figsize=(20, 18))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


plotImages(sample_training_images[:20])
plotImages(sample_validation_images[:20])

# %% Create the model
model = Sequential(
    [
        Conv2D(
            16,
            3,
            padding="same",
            activation="relu",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        ),
        MaxPooling2D(),
        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(1),
    ]
)

# Define and early stopping callback
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=30,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

# Set an optimiser with a custom learning rate
# opt = tf.keras.optimizers.Adam(learning_rate=0.2)
model.compile(
    optimizer="Adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

# %% Train the model
history = model.fit(
    train_data_gen,
    callbacks=callback,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    workers=4,
)


#%%
model.summary()

# %% Visualise training results
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(loss))

plt.figure(figsize=(8, 8))

# Subplot for accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

# Subplot for loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

# %% Evaluate training set performance
train_data = train_image_generator.flow_from_directory(
    batch_size=total_train,
    directory=train_dir,
    shuffle=False,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)


ytrue = train_data[0][1]
ypred = np.array([i[0] for i in (model.predict(train_data) > 0.5).astype("int32")])

sum(ytrue == ypred) / len(ypred)

# %% Evaluate validation set performance
val_data = validation_image_generator.flow_from_directory(
    batch_size=total_val,
    directory=validation_dir,
    shuffle=False,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)


ytrue = val_data[0][1]
ypred = np.array([i[0] for i in (model.predict(val_data) > 0.5).astype("int32")])

sum(ytrue == ypred) / len(ypred)

# %% Export the model
# Save the model
model.save("./models/classifier_v1")

# %% Test the model on a completely random picture that you chose
# testmod2 = tf.keras.models.load_model("./models/classifier_v1")
