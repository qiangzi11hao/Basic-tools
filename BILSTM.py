import tensorflow as tf


def bilstm(x, hidden_size):
    """
    :param x: [batch, height, width]   / [batch, step, embedding_size]
    :param hidden_size: lstm隐藏层节点个数
    :return: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]
    """

    input_x = tf.transpose(x, [1, 0, 2])
    input_x = tf.unpack(input_x)

    lstm_fw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                     dtype=tf.float32)
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                               dtype=tf.float32)

    output = tf.pack(output)
    output = tf.tanspose(output, [1, 0, 2])

    return output

class BILSTM(object):
    def __init__(self, hidden_size, steps, num_words,num_layers=1, dropout=0, embedding_dim=100,
                 is_training=False, embedding_matrix=None, final_unit=False):
        self.hidden_size = hidden_size
        self.num_steps = steps
        self.num_words = num_words
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.final_unit = final_unit

        self.input_x = tf.placeholder(tf.float32, [None, self.num_steps], name='input_x')

        if embedding_matrix != None:
            self.embedding = tf.Variable(embedding_matrix,
                                         trainable = False,
                                         name = 'embedding matrix',
                                         dtype = tf.float32)
        else:
            self.embedding = tf.get_variable('embedding matrix',
                                             shape = [self.num_words, self.embedding_dim],
                                             dtype = tf.float32)

        self.input_x_embed = tf.nn.embedding_lookup(self.embedding, self.input_x)

        self.input_x_embed = tf.transpose(self.input_x_embed, [1, 0, 2])
        self.input_x_embed = tf.unstack(self.input_x_embed)

        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

        if is_training:
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=(1 - self.dropout))
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=(1 - self.dropout))

        lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * self.num_layers)
        lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * self.num_layers)

        self.length = tf.reduce_sum(tf.sign(self.input_x), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        try:
            self.outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                 lstm_bw_cell,
                                                                 self.input_x_embed,
                                                                 dtype=tf.float32,
                                                                 sequence_length=self.length)
        except Exception:
            self.outputs = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                 lstm_bw_cell,
                                                                 self.input_x_embed,
                                                                 dtype=tf.float32,
                                                                 sequence_length=self.length)
        if self.final_unit:
            self.outputs = self.outputs[-1]
        else:
            self.outputs = tf.stack(self.outputs)
            self.outputs = tf.transpose(self.outputs, [1, 0, 2])

