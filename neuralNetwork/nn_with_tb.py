import tensorflow as tf
import numpy as np

# add n_layer & layer_name
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            # for View
            tf.summary.histogram(layer_name + '/weights', Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            # for View
            tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )

        # for View
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


#
# Build Neural Network ----------------------------------------
#
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    # define placeholder for inputs to network
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x_in')
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_in')

# layer 1: input layer
# hidden layer got 10 neuron
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu, n_layer=1)
# layer 2: output layer
prediction = add_layer(l1, 10, 1, activation_function=None, n_layer=2)


with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)



sess = tf.Session() # get session

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(tf.global_variables_initializer())


#
# Training ----------------------------------------
#

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # check step improvement
        rs = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(rs, i)
