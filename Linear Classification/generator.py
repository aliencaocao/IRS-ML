import tensorflow as tf

def gen(n):
    theta = (2, 3)
    theta_nought = 2000
    while True:
        X = tf.random.uniform((n,2), minval=-1000., maxval=1500., dtype=tf.float32)
        y = tf.math.sign(tf.reduce_sum(X*theta, axis=-1) - theta_nought)
        yield X, y
