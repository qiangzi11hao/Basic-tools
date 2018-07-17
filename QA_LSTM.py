import tensorflow as tf
from BILSTM import bilstm
from CNN import cnn_qa
from utils import feature2cos, get_feature, get_feature_mask, get_feature_att, cal_loss_and_acc, get_rnn2cnn_out, get_rnn2cnn_out_hxh


class LSTM_QA(object):
    def __init__(self, batch_size, steps, embeddings, embedding_size, rnn_size, num_rnn_layers,
                 max_grad_norm, filter_sizes, num_filters, self_att=True, model_choice=0,mode="qa",
                 rnn_windows=5, l2_reg_lambda=0.0, adjust_weight=False, label_weight=[], is_training=True):

        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight
        self.rnn_size = rnn_size
        self.steps = steps
        self.max_grad_norm = max_grad_norm
        self.l2_reg_lambda = l2_reg_lambda
        self.is_training = is_training
        self.filter_sizes = list(map(int, filter_sizes.split(',')))
        self.num_filters = num_filters
        self.mode_choice = model_choice
        self.rnn_windows = rnn_windows
        self.att = self_att

        self.keep_pron = tf.placeholder(tf.float32, name='keep_prob')
        self.lr = tf.Variable(0.0, trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[], name='new-learning-rate')
        self.lr_update = tf.assign(self.lr, self.new_lr)

        self.ori_input = tf.placeholder(tf.int32, shape=[None,self.steps], name='ori_inputs_quests')
        self.cand_input = tf.placeholder(tf.int32, shape=[None, self.steps], name='cand_inputs_quests')
        self.neg_input = tf.placeholder(tf.int32, shape=[None, self.steps], name='neg_inputs_quests')
        self.test_q = tf.placeholder(tf.int32, shape=[None,self.steps], name='test_q')
        self.test_a = tf.placeholder(tf.int32, shape=[None,self.steps], name='test_a')

        self.ori_q_len = tf.count_nonzero(self.ori_input, 1)
        self.cand_a_len  = tf.count_nonzero(self.cand_input, 1)
        self.neg_a_len = tf.count_nonzero(self.neg_input, 1)
        self.test_q_len = tf.count_nonzero(self.test_q, 1)
        self.test_a_len = tf.count_nonzero(self.test_a, 1)

        #embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding_layer'):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=Trur, name='W')
            ori_que = tf.nn.embedding_lookup(W, self.ori_input)
            cand_que = tf.nn.embedding_lookup(W, self.cand_input)
            neg_que = tf.nn.embedding_lookup(W, self.neg_input)
            test_q = tf.nn.embedding_lookup(W, self.test_q)
            test_a = tf.nn.embedding_lookup(W, self.test_a)


        # biLSTM
        with tf.variable_scope("LSTM_scope1", reuse=None):
            ori_q = bilstm(ori_que, self.rnn_size)

        with tf.variable_scope("LSTM_scope1", reuse=True):
            test_q = bilstm(test_q, self.rnn_size)
            test_a = bilstm(test_a, self.rnn_size)

            if mode=='qq':
                cand_a = bilstm(cand_que, self.rnn_size)
                neg_a = bilstm(neg_que, self.rnn_size)

        if mode=='qa':
            with tf.variable_scope("LSTM_scope2", reuse=None):
                cand_a = bilstm(cand_que, self.rnn_size)
            with tf.variable_scope("LSTM_scope2", reuse=True):
                neg_a = bilstm(neg_que, self.rnn_size)



