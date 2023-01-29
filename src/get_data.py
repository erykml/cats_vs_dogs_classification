import os
import tensorflow as tf
import shutil

DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
DATA_PATH = "data/raw"

path_to_zip = tf.keras.utils.get_file(
    "cats_and_dogs.zip", origin=DATA_URL, extract=True
)
download_path = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

train_dir_from = os.path.join(download_path, "train")
validation_dir_from = os.path.join(download_path, "validation")

train_dir_to = os.path.join(DATA_PATH, "train")
validation_dir_to = os.path.join(DATA_PATH, "validation")

shutil.move(train_dir_from, train_dir_to)
shutil.move(validation_dir_from, validation_dir_to)
