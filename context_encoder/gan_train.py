import sys
import tensorflow as tf
from context_encoder.model import context_encoder, get_discriminator
from context_encoder.dataset import data_pipeline
from context_encoder.utils import log_images
import keras
import wandb
import numpy as np

if __name__ == "__main__":
    tf.random.set_seed(42)

    if len(sys.argv) < 2:
        path = 'data'
    else:
        path = sys.argv[1]

    EPOCHS = 400
    BATCH_SIZE = 32
    REC_LOSS_W = 0.999
    ADV_LOSS_W = 0.001

    train = data_pipeline(path + '/train', batch_size=BATCH_SIZE)
    val = data_pipeline(path + '/val', batch_size=BATCH_SIZE, augment=False)

    X_log, y_log = next(iter(val.take(1)))

    G = context_encoder()
    G_optimizer = keras.optimizers.Adam()

    mse = keras.losses.MeanSquaredError()
    D = get_discriminator()
    D_optimizer = keras.optimizers.Adam()

    wandb.init(
        project="inpainting",
        name="context_encoder" + str(run),
        config={
            "loss": 'L2 + adversarial',
            "optimizer": 'adam',
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }
    )

    for epoch in range(EPOCHS):
        for i, (X, y) in enumerate(train):  # type: ignore
            with tf.GradientTape() as tape:
                preds = G(X)
                fake = D(preds)
                real = D(y)
                d_loss = -tf.reduce_mean(
                    tf.math.log(real) + tf.math.log(1 - fake))
            grads = tape.gradient(d_loss, D.trainable_variables)
            D_optimizer.apply_gradients(zip(grads, D.trainable_variables))

            with tf.GradientTape() as tape:
                preds = G(X)
                fake = D(preds)
                adv_loss = tf.reduce_mean(tf.math.log(fake))
                rec_loss = tf.reduce_mean(mse(preds, y))
                joint_loss = ADV_LOSS_W * adv_loss + REC_LOSS_W * rec_loss
            grads = tape.gradient(joint_loss, G.trainable_variables)
            G_optimizer.apply_gradients(zip(grads, G.trainable_variables))

            if i % 100 == 0:
                wandb.log({
                    "epoch": epoch,
                    "d_loss": d_loss,
                    "adv_loss": adv_loss,
                    "rec_loss": rec_loss,
                    "joint_loss": joint_loss,
                })

        log_images(G, X_log, y_log, epoch)

    G.save(f'models/context_encoder_GAN.h5')
