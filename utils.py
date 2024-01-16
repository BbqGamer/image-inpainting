import numpy as np
import cv2
import os
import tensorflow as tf


def mask_image(img):
    img = img.numpy()
    h = img.shape[0]
    w = img.shape[1]
    mask = np.ones((h, w), dtype=np.uint8)
    for _ in range(np.random.randint(1, 10)):
        x1, x2 = np.random.randint(0, w), np.random.randint(0, w)
        y1, y2 = np.random.randint(0, h), np.random.randint(0, h)
        thickness = np.random.randint(10, 50)
        cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),thickness)
    return cv2.bitwise_and(img, img, mask=mask)


def augment(img):
    """Creates random mask based on image shape"""
    return tf.py_function(func=mask_image, inp=[img], Tout=tf.float32)


def parse(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def get_filenames(path):
    filenames = []
    for label in os.listdir(path):
        for file in os.listdir(path + '/' + label):
            filenames.append(path + '/' + label + '/' + file)
    return filenames


def data_pipeline(path, batch_size=32):
    filenames = get_filenames(path)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.shuffle(len(filenames))
    ds = ds.map(parse, num_parallel_calls=4)
    ds = ds.map(augment, num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds

