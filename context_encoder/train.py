import tensorflow as tf
import sys
from dataset import data_pipeline
from context_encoder.model import context_encoder
import wandb
from wandb.keras import WandbCallback
import keras
import numpy as np

tf.random.set_seed(42)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))


def train(loss_fn, optimizer, path):
    EPOCHS = 400
    BATCH_SIZE = 32
    train = data_pipeline(path + '/train', batch_size=BATCH_SIZE)
    val = data_pipeline(path + '/val', batch_size=BATCH_SIZE)

    X_log, y_log = next(iter(val.take(1)))

    model = context_encoder()
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=['mae', 'mse', dice_coef])

    class ImageCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            preds = model.predict(X_log).numpy()
            original = X_log.numpy()
            predicted = original.copy()
            original[:, 96:160, 96:160] = y_log.numpy()
            predicted[:, 96:160, 96:160] = preds
            original = np.concatenate(original, axis=1)
            predicted = np.concatenate(predicted, axis=1)
            res = np.concatenate([original, predicted], axis=0)
            images = wandb.Image(res)
            wandb.log({"images": images})

    wandb.init(
        project="inpainting",
        name="context_encoder",
        config={
            "loss": loss_fn,
            "optimizer": optimizer,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }
    )

    model.fit(train, validation_data=val, epochs=EPOCHS,
              callbacks=[WandbCallback(), ImageCallback()])

    wandb.finish()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        path = 'data'
    else:
        path = sys.argv[1]

    losses = ['mse', 'mae']
    optimizers = ['adam', 'sgd', 'rmsprop']

    for loss in losses:
        for optimizer in optimizers:
            train(loss, optimizer, path)
