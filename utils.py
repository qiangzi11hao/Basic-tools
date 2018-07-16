import tensorflow as tf


def postion_embedding(inputs,  position_size):
    '''
    inputs是一个形如(batch_size, seq_len, word_size)的张量；
    函数返回一个形如(batch_size, seq_len, position_size)的位置张量。
    '''

    batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
    position_j = 1. / tf.pow(10000.,
                             2*tf.range(position_size / 2, dtype=tf.float32,
                                        ) / position_size)
    position_j = tf.expand_dims(position_j, 0)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    position_i = tf.expand_dims(position_i, 1)
    position_ij = tf.matmul(position_i, position_j)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
    position_embedding = tf.expand_dims(position_ij, 0) + tf.zeros(batch_size, seq_len, position_size)

    return position_embedding

