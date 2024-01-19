import tensorflow as tf
import sys
from dataset import data_pipeline
from model import context_encoder
from wandb.keras import WandbCallback

AUTOTUNE = tf.data.experimental.AUTOTUNE

path = sys.argv[1]

BATCH_SIZE = 32
ds = data_pipeline(path, batch_size=BATCH_SIZE)

VAL_SIZE = int(0.2 * len(ds))
train = ds.skip(VAL_SIZE)
val = ds.take(VAL_SIZE)

model = context_encoder()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

EPOCHS = 20
model.fit(train, validation_data=val, epochs=EPOCHS,
          callbacks=[WandbCallback()])
