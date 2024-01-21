import tensorflow as tf
import sys
from context_encoder.dataset import data_pipeline
from context_encoder.model import context_encoder
from context_encoder.utils import log_images
import wandb
from wandb.keras import WandbCallback
import keras
import numpy as np

tf.random.set_seed(42)


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))


def train(loss_fn, optimizer, path, run):
    EPOCHS = 400
    BATCH_SIZE = 32
    train = data_pipeline(path + '/train', batch_size=BATCH_SIZE)
    val = data_pipeline(path + '/val', batch_size=BATCH_SIZE, augment=False)

    X_log, y_log = next(iter(val.take(1)))

    model = context_encoder()
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=['mae', 'mse', dice_coef])

    class ImageCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_images(model, X_log, y_log, epoch)

    wandb.init(
        project="inpainting",
        name="context_encoder" + str(run),
        config={
            "loss": loss_fn,
            "optimizer": optimizer,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }
    )

    model.fit(train, validation_data=val, epochs=EPOCHS,
              callbacks=[WandbCallback(save_model=False), ImageCallback(),
                         keras.callbacks.EarlyStopping(patience=10)])

    model.save(f'models/context_encoder{run}.h5')

    wandb.finish()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        path = 'data'
    else:
        path = sys.argv[1]

    losses = ['mse', 'mae']
    optimizers = ['adam', 'sgd', 'rmsprop']

    run = 0
    for loss in losses:
        for optimizer in optimizers:
            if run == 0:
                run += 1
                continue
            train(loss, optimizer, path, run)
            run += 1
