import gradio as gr
import numpy as np
import time
from keras.saving import load_model
from context_encoder.model import ChannelwiseFullyConnected
from context_encoder.train import dice_coef


model = load_model('models/context_encoder3.h5',
                   custom_objects={'ChannelwiseFullyConnected': ChannelwiseFullyConnected, 'dice_coef': dice_coef})

MASK_START = 96
MASK_SIZE = 64
MASK_END = MASK_START + MASK_SIZE


def inpainting(image):
    cropped = image.copy()
    cropped[MASK_START:MASK_END, MASK_START:MASK_END, :] = 0

    print(cropped.dtype)

    cropped_input = cropped.astype('float32') / 255.0
    pred = model.predict(np.array([cropped_input]))
    print(pred.dtype)
    result = cropped.copy()
    result[MASK_START:MASK_END, MASK_START:MASK_END, :] = pred[0] * 255.0

    yield cropped, result


demo = gr.Interface(
    inpainting,
    inputs=gr.Image(label="Input", type="numpy"),
    outputs=[gr.Image(label="Cropped"), gr.Image(label="Result")],
)

demo.launch()
