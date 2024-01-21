import numpy as np
import wandb


def log_images(G, X, y, epoch):
    preds = G.predict(X)
    original = X.numpy()
    predicted = original.copy()
    original[:, 96:160, 96:160] = y.numpy()
    predicted[:, 96:160, 96:160] = preds
    original = np.concatenate(original, axis=1)
    predicted = np.concatenate(predicted, axis=1)
    res = np.concatenate([original, predicted], axis=0)
    images = wandb.Image(res)
    wandb.log({
        "epoch": epoch,
        "images": images}
    )
