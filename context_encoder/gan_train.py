import sys
import tensorflow as tf
from context_encoder.model import context_encoder, get_discriminator
from context_encoder.dataset import data_pipeline
from context_encoder.utils import log_images
import keras
import wandb
import numpy as np
import argparse


@tf.function
def train_step(X, y):
    with tf.GradientTape() as tape:
        preds = G(X)
        fake = D(preds)
        real = D(y)
        d_loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
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
    return d_loss, adv_loss, rec_loss, joint_loss


if __name__ == "__main__":
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--wandb', type=str, default='')
    parser.add_argument('--save', type=str,
                        default='models/context_encoder_GAN.h5')
    parser.add_argument('--epochs', type=int, default=400)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = 32
    REC_LOSS_W = 0.999
    ADV_LOSS_W = 0.001

    train = data_pipeline(args.data + '/train',
                          batch_size=BATCH_SIZE)
    val = data_pipeline(args.data + '/val', batch_size=BATCH_SIZE,
                        augment=False)

    X_log, y_log = next(iter(val.take(1)))

    G = context_encoder()
    G_optimizer = keras.optimizers.Adam()

    mse = keras.losses.MeanSquaredError()
    D = get_discriminator()
    D_optimizer = keras.optimizers.Adam()

    if args.wandb:
        wandb.init(
            project="inpainting",
            name=args.wandb,
            config={
                "loss": 'L2 + adversarial',
                "optimizer": 'adam',
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
            }
        )

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}/{EPOCHS}")
        for i, (X, y) in enumerate(train):  # type: ignore
            d_loss, adv_loss, rec_loss, joint_loss = train_step(X, y)

            if i % 100 == 0:
                print(
                    f"Batch {i}: d_loss: {d_loss:.4f}, adv_loss: {adv_loss:.4f}, rec_loss: {rec_loss:.4f}, joint_loss: {joint_loss:.4f}")

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "d_loss": d_loss,
                "adv_loss": adv_loss,
                "rec_loss": rec_loss,
                "joint_loss": joint_loss,
            })

            log_images(G, X_log, y_log, epoch)

    if args.wandb:
        wandb.finish()

    if args.save:
        G.save(f'models/context_encoder_GAN.h5')