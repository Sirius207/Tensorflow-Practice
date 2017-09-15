import tensorflow as tf

x1 = tf.placeholder(tf.float32)
y1 = tf.placeholder(tf.float32)
z1 = tf.multiply(x1, y1)

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
    z1_value = sess.run(z1, feed_dict={x1: [7.], y1: [2.]})  # 14
    print(z1_value)

    z2_value = sess.run(
        z2,
        feed_dict={
            x1: 1, y1: 2,
            x2: [[2], [2]], y2: [[3, 3]]
        })
    print(z2_value)
