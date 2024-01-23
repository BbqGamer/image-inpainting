import gradio as gr
import numpy as np
from keras.saving import load_model
from context_encoder.model import ChannelwiseFullyConnected
from context_encoder.train import dice_coef
from enum import Enum

MASK_START = 96
MASK_SIZE = 64
MASK_END = MASK_START + MASK_SIZE


models = [None, None, None, None]
choices = [
    'L2 MSE RMSPROP',
    'L2 MAE ADAM',
    'L2 MAE RMSPROP',
    'L2 ADV'
]
model_files = [
    'models/context_encoder2.h5',
    'models/context_encoder3.h5',
    'models/context_encoder5.h5',
    'models/context_encoder_GAN.h5'
]


def load_models(choice):
    idx = choices.index(choice)
    if models[idx] is None:
        models[idx] = load_model(model_files[idx],
                                 custom_objects={
            'ChannelwiseFullyConnected': ChannelwiseFullyConnected,
            'dice_coef': dice_coef}
        )
    return models[idx]


def inpainting(image, choice):
    model = load_models(choice)

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
    inputs=[gr.Image(label="Input", type="numpy"),
            gr.Dropdown(choices=choices, label="Model")],
    outputs=[gr.Image(label="Cropped"), gr.Image(label="Result")],
)

demo.launch()
