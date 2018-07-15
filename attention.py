import tensorflow  as tf

def Mask(inputs, seq_len, mode='mul'):
    '''
    inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
    seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
    mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
    add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
    '''
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)  # 保持首尾
        if mode == 'mul':
            return inputs*mask
        if mode == 'add':
            return inputs - (1-mask) * 1e12

def Dense(inputs, output_size, bias=True, seq_len=None):
    input_size = int(inputs.shape[-1])

    W = tf.Variable(tf.random_uniform([input_size, output_size], -0.05, 0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([output_size], -0.05, 0.05))
    else:
        b = 0

    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [output_size]], 0))

    if seq_len:
        outputs  = Mask(outputs, seq_lenm, 'mul')

    return outputs

def Attention(Q, K, V, head_num, head_size, Q_len=None, V_len=None):
    Q = Dense(Q, head_num * head_size, False)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], head_num, head_size))
    Q = tf.transpose(Q, [0, 2, 1, 3])
    K = Dense(K, head_num * head_size, False)
    K = tf.reshape(K, (-1, tf.shape(K)[1], head_num, head_size))
    K = tf.transpose(K, [0, 2, 1, 3])
    V = Dense(V, head_num * head_size, False)
    V = tf.reshape(V, (-1, tf.shape(K)[1], head_num, head_size))
    V = tf.transpose(V, [0, 2, 1, 3])

    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(head_size))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, V_len, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)

    outputs = tf.matmul(A, V)
    outputs = tf.transpose(outputs, [0, 2, 1, 3])
    outputs = tf.reshape(outputs, (-1, tf.shape(outputs)[1], head_num * head_size))
    outputs = Mask(outputs, Q_len, 'mul')

    return  outputs