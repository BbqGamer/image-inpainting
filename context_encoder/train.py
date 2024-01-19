import tensorflow as tf
import sys
from dataset import data_pipeline
from model import context_encoder
import wandb
from wandb.keras import WandbCallback
import keras
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

path = sys.argv[1]

BATCH_SIZE = 32
ds = data_pipeline(path, batch_size=BATCH_SIZE)

VAL_SIZE = int(0.2 * len(ds))
train = ds.skip(VAL_SIZE)
val = ds.take(VAL_SIZE)
to_print = val.take(1)

model = context_encoder()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

wandb.init(project="inpainting")


class ImageCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        X, y = next(iter(to_print))
        preds = model.predict(X).numpy()
        original = X.numpy()
        predicted = original.copy()
        original[:, 96:160, 96:160] = y.numpy()
        predicted[:, 96:160, 96:160] = preds
        original = np.concatenate(original, axis=1)
        predicted = np.concatenate(predicted, axis=1)
        res = np.concatenate([original, predicted], axis=0)
        images = wandb.Image(res)
        wandb.log({"images": images})


EPOCHS = 100
model.fit(train, validation_data=val, epochs=EPOCHS,
          callbacks=[WandbCallback()])
