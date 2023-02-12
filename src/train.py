from pathlib import Path
import tensorflow as tf
from dvc.api import params_show
import pandas as pd
import os
from dvclive.keras import DVCLiveCallback

# data directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = "data/raw"
train_dir = os.path.join(DATA_DIR, "train")
validation_dir = os.path.join(DATA_DIR, "validation")

# get the params
params = params_show()["train"]
IMG_WIDTH, IMG_HEIGHT = params["image_width"], params["image_height"]
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = params["batch_size"]
LR = params["learning_rate"]
N_EPOCHS = params["n_epochs"]

# get image datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)


def get_model():
    """
    Prepare the ResNet50 model for transfer learning.
    """

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ]
    )

    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.ResNet50(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(1)

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    return model


def main():

    model_path = BASE_DIR / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    model = get_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path / "model.keras", monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5),
        tf.keras.callbacks.CSVLogger("metrics.csv"),
        DVCLiveCallback(save_dvc_exp=True),
    ]

    history = model.fit(
        train_dataset,
        epochs=N_EPOCHS,
        validation_data=validation_dataset,
        callbacks=callbacks,
    )

    pd.DataFrame(history.history).to_csv("metrics.csv", index=False)


if __name__ == "__main__":
    main()
