import tensorflow as tf
import numpy as np
import generator
gen = generator.gen(1)
X, Y = [], []

for i in range(100):
    x, y = next(gen)
    X += [x]
    Y += [y]
x = np.array(X,  dtype=float)
y = np.array(Y,  dtype=float)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=1e-4))
result = model.fit(x, y, epochs=500, verbose=True)
print(f"These are the layer variables: {model.get_weights()}")
