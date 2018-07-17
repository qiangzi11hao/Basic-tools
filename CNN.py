import tensorflow as tf


def cnn_qa(ori_q, can_a, neg_a, seq_len, hidden_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
    ori_q = tf.expand_dims(ori_q, -1)
    can_a = tf.expand_dims(can_a, -1)
    neg_a = tf.expand_dims(neg_a, -1)

    pooled_q = []
    pooled_ca = []
    pooled_na = []

    for i, filter_size in enumerate(filter_sizes):
        print(filter_size)
        with tf.name_scope("conv-maxpool-%s"%filter_size):
            filter_shape = [filter_size, hidden_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), trainable=True, name="w")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), trainable=True, name='b')
            conv1 = tf.nn.conv2d(
                ori_q,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv-1')
            h = tf.nn.relu(tf.nn.bias_add(conv1, b), name='relu-1')
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, seq_len - filter_size+1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool-1')
            pooled_q.append(pooled)

            conv2 = tf.nn.conv2d(
                can_a,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv-2')
            h = tf.nn.relu(tf.nn.bias_add(conv2, b), name='relu-2')
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, seq_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool-2')
            pooled_ca.append(pooled)

            conv3 = tf.nn.conv2d(
                neg_a,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv-3')
            h = tf.nn.relu(tf.nn.bias_add(conv3, b), name='relu-3')
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, seq_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool-3')
            pooled_na.append(pooled)

    num_total_filters = len(filter_sizes) * num_filters
    pooled_q = tf.concat(pooled_q, 3)
    output_q = tf.reshape(pooled_q, [-1, num_total_filters])
    pooled_ca = tf.concat(pooled_ca, 3)
    output_ca = tf.reshape(pooled_ca, [-1, num_total_filters])
    pooled_na = tf.concat(pooled_na, 3)
    output_na = tf.reshape(pooled_na, [-1, num_total_filters])

    return output_q, output_ca, output_na




class CNN(object):
    """
    A sample CNN for classification
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """


def __init__(self, sequence_length, num_classes, vocab_size,
             embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

    #定义输入输出以及dropout
    self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
    self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
    self.dropout = tf.placeholder(tf.float32, name='dropout_keep_prob')

    #设置l2 loss
    l2_loss = tf.constant(0.0)

    #embedding 层
    with tf.device('/cpu:0'), tf.name_scope('embedding'):
        self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                             name='W')
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)

    pooled_ouputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s"%filter_size):
            #convolution layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters], name='b'))
            conv = tf.nn.conv2d(self.embedded_chars_expand,
                                W,
                                strides=[1,1,1,1],
                                padding='VALID',
                                name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv,b), name='relu')

            pooled = tf.nn.max_pool(h,
                                    ksize=[1, sequence_length-filter_size+1, 1, 1],
                                    strides=[1,1,1,1],
                                    name='pool')
            pooled_ouputs.append(pooled)

    num_filters_total = num_filters*len(filter_sizes)
    self.h_pool = tf.concat(pooled_ouputs, 3)
    self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

    #dropout
    with tf.name_scope('dropout'):
        self.h_drop = tf.nn.dropout((self.h_pool_flat, self.dropout))

    #Final scores
    with tf.name_scope("output"):
        W = tf.get_variable(
            "W",
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
        self.predictions = tf.argmax(self.scores, 1, name='predictions')

    #计算交叉熵
    with tf.name_scope('loss'):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
        self.loss = tf.reduce_mean(losses) + l2_loss*l2_reg_lambda

    #acc
    with tf.name_scope('accuracy'):
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

