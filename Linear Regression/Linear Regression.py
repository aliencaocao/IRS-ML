import tensorflow as tf
import generator
gen = generator.gen(1)


class LinearModel:
    def __call__(self, x):
        return self.Weight * x + self.Bias

    def __init__(self):
        self.Weight = tf.Variable(1.0)
        self.Bias = tf.Variable(1.0)


def loss(y, pred):  # Mean Squared Error
    return tf.reduce_mean(tf.square(y - pred))


def train(linear_model, x, y, lr):
    with tf.GradientTape() as t:
        current_loss = loss(y, linear_model(x))

    _weight, _bias = t.gradient(current_loss, [linear_model.Weight, linear_model.Bias])
    linear_model.Weight.assign_sub(lr * _weight)
    linear_model.Bias.assign_sub(lr * _bias)


linear_model = LinearModel()
Weights, Biases = [], []
epochs = 1000
for epoch_count in range(epochs):
    x, y = next(gen)
    Weights.append(linear_model.Weight.numpy())
    Biases.append(linear_model.Bias.numpy())
    real_loss = loss(y, linear_model(x))
    train(linear_model, x, y, lr=1e-5)
    print(f"Epoch count {epoch_count}: Loss value: {real_loss.numpy()}")

print(f'Weight: {linear_model.Weight.numpy()}, Bias: {linear_model.Bias.numpy()}')
