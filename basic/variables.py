import tensorflow as tf


# variable in global_variable
state = tf.Variable(0, name='counter')

one = tf.constant(1)

# state = state + 1
add_operation = tf.add(state, one)
# or simply use add_operation = tf.add(state, 1)

# state = new_value
update_operation = tf.assign(state, add_operation)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update_operation)
        print(sess.run(state))
