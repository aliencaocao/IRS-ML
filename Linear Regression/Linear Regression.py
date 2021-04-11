# Linear Regression project with generalized input dimensions and L2 normalisation
__author__ = "Billy Cao"
import tensorflow as tf
import generator

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)  # change to false when running on ML server
tf.keras.mixed_precision.set_global_policy('mixed_float16')

batchSize = 1
gen = generator.gen2d(batchSize)  # gen2d for 2D input
epochs = 1000
regTerm = 0.001
learning_rate = 1e-5


class LinearModel:  # initializing to 1 now, but also can do 0 or random
    def __call__(self, x):  # predicting function
        return self.Weight * x + self.Bias

    def __init__(self):
        self.Weight = tf.Variable(1.0, shape=tf.TensorShape(None))  # initialize m to any shape
        self.Bias = tf.Variable(1.0)


def loss(y, pred):  # Mean Squared Error with L2 Normalisation
    return tf.reduce_mean(tf.square(y - pred)) + tf.reduce_sum(regTerm * tf.square(linear_model.Weight))


def train(linear_model, x, y, lr):
    # use to reshape into 2D vector if input is of unknown dimension
    # if len(x.shape) == 1:
    #     X = tf.reshape(x, [x.shape[0], 1])
    with tf.GradientTape(persistent=False) as t:  # persistent=True is needed if assigning dy_dWeight, dy_dBias in 2 lines. Limits the times u can call it to once
        current_loss = loss(y, linear_model(x))
    dy_dWeight, dy_dBias = t.gradient(current_loss, [linear_model.Weight, linear_model.Bias])
    linear_model.Weight.assign_sub(lr * dy_dWeight)
    linear_model.Bias.assign_sub(lr * dy_dBias)


linear_model = LinearModel()
sampleX, sampleY = next(gen)
linear_model.Weight.assign([1.0] * sampleX.shape[-1])  # initialize m to 1.0 and make it same dimension as the input
for epoch_count in range(epochs):
    x, y = next(gen)
    real_loss = loss(y, linear_model(x))
    train(linear_model, x, y, lr=learning_rate)
    print(f"Epoch count {epoch_count}: Loss value: {real_loss.numpy()}")

print(f'Weight: {linear_model.Weight.numpy()}, Bias: {linear_model.Bias.numpy()}')
