import keras
import tensorflow as tf
from keras import layers
from keras import activations
from context_encoder.common import ChannelwiseFullyConnected

BOTTLENECK_SIZE = 1000
BOTTLENECK_SHAPE = (8, 8, BOTTLENECK_SIZE)


def encoder(input_shape=(256, 256, 3)):
    encoder = keras.models.Sequential(name='encoder')
    encoder.add(layers.Input(input_shape))
    for filters in [64, 64, 128, 256, 512]:
        encoder.add(layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same'
        ))
        encoder.add(layers.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(layers.Dense(BOTTLENECK_SIZE, activation='relu'))
    assert encoder.output_shape[1:] == BOTTLENECK_SHAPE
    return encoder


def decoder(input_shape=BOTTLENECK_SHAPE):
    decoder = keras.models.Sequential(name='decoder')
    decoder.add(layers.Input(input_shape))
    decoder.add(ChannelwiseFullyConnected(512))

    for filters in [512, 256, 128]:
        decoder.add(layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same'
        ))
        decoder.add(layers.UpSampling2D(size=(2, 2)))
    decoder.add(layers.Conv2D(
        64, (3, 3), activation='relu', padding='same'))
    decoder.add(layers.Conv2D(
        3, (3, 3), activation='sigmoid', padding='same'))
    return decoder


def context_encoder(input_shape=(256, 256, 3)):
    encoder_model = encoder(input_shape)
    decoder_model = decoder()
    model = keras.models.Sequential(name='context_encoder')
    model.add(layers.Input(input_shape))
    model.add(encoder_model)
    model.add(decoder_model)
    return model


def get_discriminator():
    discriminator = keras.models.Sequential(name='discriminator')
    discriminator.add(layers.Input((64, 64, 3)))
    discriminator.add(layers.MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(layers.Conv2D(
        64, (3, 3), activation='relu', padding='same'))
    discriminator.add(layers.MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(layers.Conv2D(
        128, (3, 3), activation='relu', padding='same'))
    discriminator.add(layers.MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(layers.Conv2D(
        128, (3, 3), activation='relu', padding='same'))
    discriminator.add(layers.MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(256, activation='relu'))
    discriminator.add(layers.Dense(1, activation='sigmoid'))
    return discriminator


def getUnet(input_shape=(256, 256, 3), output_channels=3):
    inputs = tf.keras.Input(shape=input_shape)
    x1 = layers.Conv2D(32, 4, strides=2, padding='same',
                       activation='relu')(inputs)
    x2 = layers.Conv2D(64, 4, strides=2, padding='same',
                       activation='relu')(x1)
    x3 = layers.Conv2D(64, 4, strides=2, padding='same',
                       activation='relu')(x2)
    x4 = layers.Conv2D(128, 4, strides=2, padding='same',
                       activation='relu')(x3)
    x5 = layers.Conv2D(256, 4, strides=2, padding='same',
                       activation='relu')(x4)

    x6 = layers.Conv2D(512, 4, strides=2, padding='same',
                       activation='relu')(x5)

    bottleneck = ChannelwiseFullyConnected(1024)(x6)

    # Decoder
    x7 = layers.Conv2DTranspose(
        256, 4, strides=2, padding='same', activation='relu')(bottleneck)
    x8 = layers.Concatenate()([x7, x5])
    x9 = layers.Conv2DTranspose(
        128, 4, strides=2, padding='same', activation='relu')(x8)
    x10 = layers.Concatenate()([x9, x4])
    x11 = layers.Conv2DTranspose(
        128, 4, strides=2, padding='same', activation='relu')(x10)
    x12 = layers.Concatenate()([x11, x3])
    x13 = layers.Conv2DTranspose(
        64, 4, strides=2, padding='same', activation='relu')(x12)
    x13 = layers.Concatenate()([x13, x2])

    outputs = layers.Conv2D(
        output_channels, 3, padding='same', activation='sigmoid')(x13)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')
    return model


if __name__ == '__main__':
    model = getUnet()
    model.summary(expand_nested=True)

    import time
    # feed forward time test
    input = tf.random.normal((1, 256, 256, 3))
    start = time.time()
    output = model(input)
    end = time.time()
    print("Took {:.2f} seconds".format(end - start))

    D = get_discriminator()
    D.summary()
