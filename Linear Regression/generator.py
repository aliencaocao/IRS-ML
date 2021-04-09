import tensorflow as tf
import random

def gen2d(batch_size):
    m = tf.convert_to_tensor([1, 2], dtype=tf.float32)
    c = 23.23
    while True:
        X = tf.random.uniform((batch_size, 2), minval=-100., maxval=100., dtype=tf.float32)
        y = X*m + c

        yield tf.convert_to_tensor(X), tf.convert_to_tensor(y)

def gen(batch_size):
    m = tf.convert_to_tensor(2356.231, dtype=tf.float32)
    c = 2.23
    while True:
        X = tf.random.uniform((batch_size,), minval=-100., maxval=100., dtype=tf.float32)
        y = X*m + c + random.randint(-100, 100)

        yield tf.convert_to_tensor(X), tf.convert_to_tensor(y)
