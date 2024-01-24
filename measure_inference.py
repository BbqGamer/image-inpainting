from context_encoder.dataset import data_pipeline
import keras
import time
from context_encoder.model import ChannelwiseFullyConnected
from context_encoder.train import dice_coef
from collections import defaultdict

if __name__ == '__main__':
    MODELS = ['context_encoder2.h5', 'context_encoder3.h5',
              'context_encoder5.h5', 'context_encoder_GAN.h5']

    results = defaultdict(list)
    for modelf in MODELS:
        ds = data_pipeline('data/test', cache=False,
                           shuffle=False, augment=False)
        length = len(ds) * 32
        model = keras.models.load_model('models/' + modelf, custom_objects={
                                        'ChannelwiseFullyConnected': ChannelwiseFullyConnected, 'dice_coef': dice_coef})
        for X, y in ds:
            start = time.time()
            model.predict(X)
            end = time.time()
            results[modelf].append(end - start)

    for k, v in results.items():
        res = sum(v)
        print(
            f"{k} took {round(res, 4)} seconds to process {length} images (its {round(res/length, 4)}) seconds per image)")
