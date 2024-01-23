import keras
import tensorflow as tf
from keras import layers, activations


class ChannelwiseFullyConnected(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(ChannelwiseFullyConnected, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            "kernel", (input_dim, self.units), initializer="glorot_uniform", trainable=True)
        super(ChannelwiseFullyConnected, self).build(input_shape)

    def call(self, inputs):
        # Reshape the input tensor to treat each channel as an independent feature
        input_shape = tf.shape(inputs)
        flattened_inputs = tf.reshape(
            inputs, [input_shape[0], -1, input_shape[-1]])  # type: ignore

        # Perform standard fully connected operation for each channel independently
        outputs = tf.matmul(flattened_inputs, self.kernel)

        # Reshape the output back to the original shape
        output_shape = tf.concat(
            [input_shape[:-1], [self.units]], axis=0)  # type: ignore
        outputs = tf.reshape(outputs, output_shape)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))
