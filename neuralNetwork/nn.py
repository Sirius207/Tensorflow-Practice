import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weight) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs

#
# Build Neural Network ----------------------------------------
#

# [[-1, -0.x,...,1]]
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# layer 1: input layer
# hidden layer got 10 neuron
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# layer 2: output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# loss = ((p - y)^2)/m
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# when 'x' is [[1., 1.]
#         [2., 2.]]
# reduction_indice = [1.,  2.]
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


# Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#
# Training ----------------------------------------
#

# for i in range(1000):
#     # training
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

#     if i % 50 == 0:
#         # check step improvement
#         print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


#
# Visualization
#

fig = plt.figure()
ax =  fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
# for continue showing
plt.ion()
plt.show()

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        # check step improvement
        predict_value = sess.run(prediction, feed_dict={xs: x_data})
        # 'r-' = red, lw = lineWidth
        lines = ax.plot(x_data, predict_value, 'r-', lw=5)
        plt.pause(0.1)
