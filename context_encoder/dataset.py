import numpy as np
import cv2
import os
import tensorflow as tf
import sys

AUTOTUNE = tf.data.experimental.AUTOTUNE

MASK_START = 96
MASK_SIZE = 64
MASK_END = MASK_START + MASK_SIZE

mask = np.ones((256, 256, 3))
mask[MASK_START:MASK_END, MASK_START:MASK_END] = 0
mask = tf.convert_to_tensor(mask, dtype=tf.float32)


def get_masked_image(image):
    """Draw 128x128 bounding box on the image using only tf operations"""
    X = image * mask
    y = image[MASK_START:MASK_END, MASK_START:MASK_END]
    y = tf.ensure_shape(y, (MASK_SIZE, MASK_SIZE, 3))
    return X, y


def shapes(masked, mask, imgs):
    masked = tf.ensure_shape(masked, (256, 256, 3))
    mask = tf.ensure_shape(mask, (256, 256))
    img = tf.ensure_shape(imgs, (256, 256, 3))
    return (masked, mask), img


def parse(filename):
    image_string = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(image_string)
    img = tf.image.convert_image_dtype(img, tf.float32)
    resized = tf.image.resize(img, (256, 256))
    return resized


def get_filenames(path):
    filenames = []
    for label in os.listdir(path):
        for file in os.listdir(path + '/' + label):
            filenames.append(path + '/' + label + '/' + file)
    return filenames


def augment_data(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image


def data_pipeline(path, batch_size=32, cache=True, shuffle=True, augment=True, small=False):
    # https://cs230.stanford.edu/blog/datapipeline/#goals-of-this-tutorial
    filenames = get_filenames(path)
    if small:
        filenames = filenames[:32]
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        ds = ds.shuffle(len(filenames), reshuffle_each_iteration=False)
    ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_data, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.map(get_masked_image, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    return ds


def imshow(image, title='img'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Demonstration of pipeline usage
    if len(sys.argv) < 2:
        path = 'data/train'
    else:
        path = sys.argv[1]

    tf.random.set_seed(42)
    ds = data_pipeline(path)
    iterator = ds.as_numpy_iterator()
    X, y = next(iterator)  # type: ignore
    Xs = np.concatenate(X[:7], axis=1)
    ys = np.concatenate(y[:7], axis=1)
    imshow(Xs)
    imshow(ys)
