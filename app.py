import tensorflow as tf
import gradio as gr
import numpy as np
from keras.saving import load_model
from context_encoder.common import ChannelwiseFullyConnected, dice_coef
print("Loading tensorflow...")
print(tf.__version__)

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
    if image.shape != (256, 256, 3):
        image = tf.image.resize(image, (256, 256)).numpy()

    model = load_models(choice)
    cropped = image.copy()
    cropped[MASK_START:MASK_END, MASK_START:MASK_END, :] = 0
    cropped_input = cropped / 255.0
    pred = model.predict(np.array([cropped_input]))
    print(pred.dtype)
    result = cropped.copy()
    result[MASK_START:MASK_END, MASK_START:MASK_END, :] = pred[0] * 255.0

    yield cropped.astype(np.uint8), result.astype(np.uint8)


if __name__ == "__main__":
    demo = gr.Interface(
        inpainting,
        inputs=[gr.Image(height=256, width=256, label="Input", type="numpy"),
                gr.Dropdown(choices=choices, label="Model", value=choices[1])],
        outputs=[
            gr.Image(height=256, width=256, label="Cropped"),
            gr.Image(height=256, width=256, label="Result")
        ],
    )

    print("Starting server...")
    demo.launch(server_name='0.0.0.0')
