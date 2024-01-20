import keras
import tensorflow as tf

BOTTLENECK_SIZE = 1000
BOTTLENECK_SHAPE = (8, 8, BOTTLENECK_SIZE)


class ChannelwiseFullyConnected(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(ChannelwiseFullyConnected, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            "kernel", (input_dim, self.units), initializer="glorot_uniform", trainable=True)
        super(ChannelwiseFullyConnected, self).build(input_shape)

    def call(self, inputs):
        # Reshape the input tensor to treat each channel as an independent feature
        input_shape = tf.shape(inputs)
        flattened_inputs = tf.reshape(
            inputs, [input_shape[0], -1, input_shape[-1]])

        # Perform standard fully connected operation for each channel independently
        outputs = tf.matmul(flattened_inputs, self.kernel)

        # Reshape the output back to the original shape
        output_shape = tf.concat([input_shape[:-1], [self.units]], axis=0)
        outputs = tf.reshape(outputs, output_shape)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)


def encoder(input_shape=(256, 256, 3)):
    encoder = keras.models.Sequential(name='encoder')
    encoder.add(keras.layers.Input(input_shape))
    for filters in [64, 64, 128, 256, 512]:
        encoder.add(keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same'
        ))
        encoder.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(keras.layers.Dense(BOTTLENECK_SIZE, activation='relu'))
    assert encoder.output_shape[1:] == BOTTLENECK_SHAPE
    return encoder


def decoder(input_shape=BOTTLENECK_SHAPE):
    decoder = keras.models.Sequential(name='decoder')
    decoder.add(keras.layers.Input(input_shape))
    decoder.add(ChannelwiseFullyConnected(512))

    for filters in [512, 256, 128]:
        decoder.add(keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same'
        ))
        decoder.add(keras.layers.UpSampling2D(size=(2, 2)))
    decoder.add(keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same'))
    decoder.add(keras.layers.Conv2D(
        3, (3, 3), activation='sigmoid', padding='same'))
    return decoder


def context_encoder(input_shape=(256, 256, 3)):
    encoder_model = encoder(input_shape)
    decoder_model = decoder()
    model = keras.models.Sequential(name='context_encoder')
    model.add(keras.layers.Input(input_shape))
    model.add(encoder_model)
    model.add(decoder_model)
    return model


def get_discriminator():
    discriminator = keras.models.Sequential(name='discriminator')
    discriminator.add(keras.layers.Input((64, 64, 3)))
    discriminator.add(keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same'))
    discriminator.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(keras.layers.Conv2D(
        128, (3, 3), activation='relu', padding='same'))
    discriminator.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(keras.layers.Conv2D(
        256, (3, 3), activation='relu', padding='same'))
    discriminator.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same'))
    discriminator.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(256, activation='relu'))
    discriminator.add(keras.layers.Dense(1, activation='sigmoid'))
    return discriminator


if __name__ == '__main__':
    model = context_encoder()
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
