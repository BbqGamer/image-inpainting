import keras
import tensorflow as tf
from utils import data_pipeline
import numpy as np
from model import vanilla_autoencoder
import wandb


def reconstruction_loss(masked, mask, y_true):
    y_true_flat = tf.reshape(y_true, [-1])
    masked_flat = tf.reshape(masked, [-1])
    mask_flat = tf.reshape(mask, [-1])

    # Only consider the pixels that are in mask
    masked_pixels = tf.where(tf.equal(mask_flat, 1))
    y_true_masked = tf.gather(y_true_flat, masked_pixels)
    masked_flat = tf.gather(masked_flat, masked_pixels)

    return tf.reduce_mean(
        tf.square(y_true_masked - masked_flat)
    )


if __name__ == '__main__':
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.02

    train = data_pipeline('data/images/train', batch_size=BATCH_SIZE)
    data_iterator = train.as_numpy_iterator()

    distorted = keras.layers.Input((256, 256, 3))
    masks = keras.layers.Input((256, 256, 1))

    autoencoder = vanilla_autoencoder()
    autoencoder = autoencoder(distorted) 
    model = keras.models.Model(inputs=distorted, outputs=autoencoder)

    adam = keras.optimizers.Adam()
    model.compile(optimizer=adam)
    model.summary()

    wandb.init(
        project="inpainting",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }
    )    

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        # Mini-batch training
        for batch in range(len(train)):
            (masked, mask), img = next(data_iterator)
            print(masked.shape, mask.shape, img.shape)

            with tf.GradientTape() as tape:
                y_pred = model(masked)
                loss = reconstruction_loss(masked, mask, y_pred)

            # Compute gradients and update weights
            grads = tape.gradient(loss, model.trainable_variables)
            adam.apply_gradients(zip(grads, model.trainable_variables))
            print(f'Batch {batch+1}/{len(train)}, loss: {loss.numpy()}')
            if batch % 100 == 0:
                wandb.log({"loss": loss.numpy()})
                preds = np.concatenate(y_pred.numpy()[:6], axis=1)
                maskeds = np.concatenate(masked[:6], axis=1)
                res = np.concatenate([preds, maskeds], axis=0)
                images = wandb.Image(res)
                wandb.log({"images": images})
    
    PATH = 'saved_model'
    model.save(PATH)
    wandb.log_model(PATH, name='autoencoder')
    wandb.finish()
