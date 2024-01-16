import numpy as np
import cv2
import os
import tensorflow as tf


def mask_image(img):
    """Creates random mask based on image shape and returns masked images
       mask consists of 1 to 10 shapes (rectangles, circles, lines)
    """
    img = img.numpy()
    height, width, _ = img.shape
    mask = np.ones((height, width), dtype=np.uint8)
    for _ in range(np.random.randint(1, 10)):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        shape = np.random.randint(0, 3)
        if shape == 0:
            w = np.random.randint(width // 8, width // 4)
            h = np.random.randint(height // 8, height // 4)
            cv2.rectangle(mask, (x - w//2, y - h//2), (x+w, y+h), (0, 0, 0), -1)
        elif shape == 1:
            r = np.random.randint(height // 8, height // 4) // 2
            cv2.circle(mask, (x, y), r, (0, 0, 0), -1)
        else:
            x2 = np.random.randint(0, width)
            y2 = np.random.randint(0, height)
            thickness = np.random.randint(width // 16, width // 8)
            cv2.line(mask, (x, y), (x2, y2), (0, 0, 0), thickness)
    return cv2.bitwise_and(img, img, mask=mask), 1 - mask, img


def augment(img):
    """Creates random mask based on image shape"""
    return tf.py_function(func=mask_image, inp=[img], Tout=[tf.float32, tf.uint8, tf.float32])


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
    # https://cs230.stanford.edu/blog/datapipeline/#goals-of-this-tutorial
    filenames = get_filenames(path)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.shuffle(len(filenames))
    ds = ds.map(parse, num_parallel_calls=4)
    ds = ds.map(augment, num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds


def imshow(image, title='img'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Demonstration of pipeline usage
    tf.random.set_seed(42)
    ds = data_pipeline('data/images/train')
    iterator = ds.as_numpy_iterator()
    batch = next(iterator)
    masked, masks, imgs = batch # type: ignore
    maskedc = np.concatenate(masked[:6], axis=1) 
    masksc = cv2.cvtColor(np.concatenate(masks[:6], axis=1), cv2.COLOR_GRAY2BGR)
    imgsc = np.concatenate(imgs[:6], axis=1)
    imshow(np.concatenate([maskedc, masksc, imgsc], axis=0), '6 examples from first batch')

