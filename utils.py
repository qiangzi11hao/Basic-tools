import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes


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


def feature2cos(f_q, f_a):
    norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(f_q, f_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(f_a, f_a), 1))
    mul_q_a = tf.reduce_sum(tf.multiply(f_q, f_a), 1)
    cos_sim = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))

    return cos_sim


def max_pooling(lstm_out):
    """
    :param lstm_out: [batch, step, rnn_size*2]
    :return:[batch, rnn_size*2]
    """
    h, w = tf.shape(lstm_out)[1], tf.shape(lstm_out)[2]

    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(
        lstm_out,
        ksize=[1, h, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')
    output = tf.reshape(output, [-1, w])

    return output


def avg_pooling(lstm_out):
    h, w = tf.shape(lstm_out)[1], tf.shape(lstm_out)[2]

    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.avg_pool(
        lstm_out,
        ksize=[1, h, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    output = tf.reshape(output, [-1, w])

    return output

def cal_loss_and_acc(ori_cand, ori_neg):
    zero = tf.fill(tf.shape(ori_cand), 0.0)
    margin = tf.fill(tf.shape(ori_cand), 0.1)
    with tf.name_scope("loss"):
        losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(ori_cand, ori_neg)))
        loss = tf.reduce_sum(losses)

    with tf.name_scope("acc"):
        correct = tf.equal(zero, losses)
        acc = tf.reduce_sum(tf.cast(correct, 'float'), name='acc')
    return  loss, acc

def mask(inputs, seq_len, max_seq=None, mode='mul'):
    '''
    inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
    seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
    mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
    add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
    '''
    if seq_len == None:
        return inputs
    else:
        mask = tf.sequence_mask(lengths=seq_len, maxlen=max_seq, dtype=dtypes.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)  # 保持[batch_size, seq_len,...]与前两维对应,后面的所有都进行mask
        if mode == 'mul':
            return inputs*mask
        if mode == 'add':
            return inputs - (1-mask) * 1e12


def get_feature(input_q, input_a):
    output_q = max_pooling(input_q)
    output_a = max_pooling(input_a)

    return output_q, output_a


def get_feature_mask(input_q, input_a, seq_q_len, seq_a_len, max_len):
    output_q = max_pooling(mask(input_q, seq_q_len, max_len))
    output_a = max_pooling(mask(input_a, seq_a_len, max_len))

    return output_q, output_a


def get_feature_att(input, att, seq_len, max_seq_len):
    dim = tf.shape(input)[2]
    att_input = tf.reshape(input, [-1, dim])
    att_e = tf.matmul(tf.tanh(tf.matmul(att_input, att['att_w']) + att['att_b']), att['att_u'])
    att_e = tf.reshape(att_e, [-1, max_seq_len])
    att_e = mask(att_e, seq_len, max_seq_len, mode='add')
    att_weights = tf.nn.softmax(att_e)
    output = tf.reduce_sum(input * array_ops.expand_dims(att_weights, 2), axis=1)

    return output


def get_door_output(input, input_door, change_input, door_w, max_seq):
    fea_size = tf.shape(input)[-1]
    door = tf.reshape(tf.matmul(input_door, door_w['d_w']) + door_w['d_b'], [-1, max_seq, fea_size])
    door = tf.sigmoid(door)
    print(door.shape)
    output_door = change_input * door + input * (1.0-door)

    return output_door


def get_rnn2cnn_out(input_q, input_a, door_w, change_w, max_seq):
    embedding_dim = tf.shape(input_q)[2]
    input_q_door = tf.reshape(input_q, [-1, embedding_dim])
    input_a_door = tf.reshape(input_a, [-1, embedding_dim])
    change_out_q = tf.reshape(tf.tanh(tf.matmul(input_q_door, change_w['c_w']) + change_w['c_b']), [-1, max_seq, embedding_dim])
    change_out_a = tf.reshape(tf.tanh(tf.matmul(input_a_door, change_w['c_w']) + change_w['c_b']), [-1, max_seq, embedding_dim])
    door_out_q = get_door_output(input_q, input_q_door, change_out_q, door_w, max_seq)
    door_out_a = get_door_output(input_a, input_a_door, change_out_a, door_w, max_seq)

    return door_out_q, door_out_a


def get_rnn2cnn_out_hxh(input_q_emb, input_q, input_a_emb, input_a, door_w, change_w, max_seq):
    embedding_dim = tf.shape(input_q)[2]
    input_q_door = tf.reshape(input_q, [-1, embedding_dim])
    input_q_emb_door = tf.reshape(input_q_emb, [-1, tf.shape(input_q_emb)[-1]])
    input_a_door = tf.reshape(input_a, [-1, embedding_dim])
    input_a_emb_door  = tf.reshape(input_a_emb, [-1, tf.shape(input_a_emb)[-1]])
    change_out_q = tf.reshape(tf.tanh(tf.matmul(input_q_door, change_w['c_w']) + change_w['c_b']), [-1, max_seq, embedding_dim])
    change_out_a = tf.reshape(tf.tanh(tf.matmul(input_a_door, change_w['c_w']) + change_w['c_b']), [-1, max_seq, embedding_dim])
    door_out_q = get_door_output(input_q, input_q_emb_door, change_out_q, door_w, max_seq)
    door_out_a = get_door_output(input_a, input_a_emb_door, change_out_a, door_w, max_seq)

    return [door_out_q, door_out_a]