import keras

def vanilla_autoencoder(input_shape=(256,256,3)):
    encoder = keras.models.Sequential(name='encoder')
    encoder.add(keras.layers.Input(input_shape))
    for filters in [32, 64, 128, 256]:
        conv, pool = DownConvBlock(filters)
        encoder.add(conv)
        encoder.add(pool)
    
    decoder = keras.models.Sequential(name='decoder')
    decoder.add(keras.layers.Input(encoder.output_shape[1:]))
    for filters in [256, 128, 64, 32]:
        pool, conv = UpConvBlock(filters)
        decoder.add(pool)
        decoder.add(conv)
    decoder.add(keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    return keras.models.Sequential([encoder, decoder], name='autoencoder')
    

def DownConvBlock(filters, shape=(3,3), activation='relu', padding='same'):
    conv = keras.layers.Conv2D(
        filters, shape, activation=activation, padding=padding
    )
    pool = keras.layers.MaxPooling2D(pool_size=(2, 2))
    return conv, pool 


def UpConvBlock(filters, shape=(3,3), activation='relu', padding='same'):
    pool = keras.layers.UpSampling2D(size=(2, 2))
    conv = keras.layers.Conv2D(
        filters, shape, activation=activation, padding=padding
    )
    return pool, conv

