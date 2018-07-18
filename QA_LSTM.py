import tensorflow as tf
from BILSTM import bilstm
from CNN import cnn_qa
from utils import feature2cos, mask, get_feature, get_feature_mask, get_feature_att, cal_loss_and_acc, get_rnn2cnn_out, get_rnn2cnn_out_hxh


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
            test_que = tf.nn.embedding_lookup(W, self.test_q)
            test_ans = tf.nn.embedding_lookup(W, self.test_a)


        # biLSTM
        with tf.variable_scope("LSTM_scope1", reuse=None):
            ori_q = bilstm(ori_que, self.rnn_size)

        with tf.variable_scope("LSTM_scope1", reuse=True):
            test_q = bilstm(test_que, self.rnn_size)
            test_a = bilstm(test_ans, self.rnn_size)

            if mode=='qq':
                cand_a = bilstm(cand_que, self.rnn_size)
                neg_a = bilstm(neg_que, self.rnn_size)

        if mode=='qa':
            with tf.variable_scope("LSTM_scope2", reuse=None):
                cand_a = bilstm(cand_que, self.rnn_size)
            with tf.variable_scope("LSTM_scope2", reuse=True):
                neg_a = bilstm(neg_que, self.rnn_size)



        ori_q = mask(ori_q, self.ori_q_len, self.steps)
        cand_a = mask(cand_a, self.cand_a_len, self.steps)
        neg_a = mask(neg_a, self.neg_a_len, self.steps)
        test_q = mask(test_q, self.test_q_len, self.steps)
        test_a = mask(test_a, self.test_a_len, self.steps)

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s"%filter_size):
                if self.mode_choice == 3:
                    filter_size = [filter_size, self.embedding_size, 1, self.num_filters]
                else:
                    filter_size = [filter_size, self.rnn_size*2, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_size, stddev=0.1), name='Kernel_W')
                b =tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='Kernel_b')
                self.kernel.append((W,b))
        in_dim = ori_que.get_shape()[2]
        with tf.variable_scope('door'):
            if self.mode_choice == 5 or self.mode_choice==6:
                door_w = {
                    'd_w': tf.get_variable('d_w', [100, 300]),
                    'd_b': tf.get_variable('d_b', [1,300]),
                }
            else:
                door_w = {
                    'd_w': tf.get_variable('d_w', [in_dim, in_dim]),
                    'd_b': tf.get_variable('d_b', [in_dim, in_dim]),
                }

        with tf.variable_scope('change'):
            if self.mode_choice==5 or self.mode_choice==6:
                change_w = {
                    'c_w': tf.get_variable('c_w', [300, 300]),
                    'c_b': tf.get_variable('c_b', [1, 300]),
                }
            else:
                change_w = {
                    'c_w': tf.get_variable('c_w', [in_dim, in_dim]),
                    'c_b': tf.get_variable('c_b', [1, in_dim]),
                }

        with tf.variable_scope('self_att'):
            self_att = {
                'att_w': tf.get_variable('att_w', [300, 300]),
                'att_b': tf.get_variable('att_b', [1, 300]),
                'att_u': tf.get_variable('att_u', [300, 1])
            }

        if self.mode_choice == 0: ##biLSTM + mask + highway+cnn_qa
            ori_q_highway, cand_a_highway = get_rnn2cnn_out(ori_q, cand_a, door_w, change_w, self.steps)
            _, neg_a_highway = get_rnn2cnn_out(ori_q, neg_a, door_w, change_w, self.steps)
            test_q_highway, test_a_highway = get_rnn2cnn_out(test_q, test_a, door_w, change_w, self.steps)
            print(ori_q_highway.shape)

            ori_q_fea, cand_a_fea, neg_a_fea = cnn_qa(ori_q=ori_q_highway, can_a=cand_a_highway,neg_a=neg_a_highway,
                                                      seq_len=self.steps,
                                                      hidden_size=2*self.rnn_size,
                                                      filter_sizes=self.filter_sizes,
                                                      num_filters=self.num_filters)
            test_q_fea, test_a_fea, _ = cnn_qa(ori_q=test_q_highway, cand_a=test_a_highway, neg_a=neg_a_highway,
                                               seq_len=self.steps,
                                               hidden_size=2*self.rnn_size,
                                               filter_sizes=self.filter_sizes,
                                               num_filters=self.num_filters)

        if self.mode_choice == 1:##biLSTM + mask + max_pooling
            ori_q_fea, cand_a_fea = get_feature_mask(ori_q, cand_a, self.ori_q_len, self.cand_a_len, self.steps)
            _, neg_a_fea= get_rnn2cnn_out(ori_q, neg_a, self.ori_q_len, self.neg_a_len, self.steps)
            test_q_fea, test_a_fea = get_feature_mask(test_q, test_a, self.test_q_len, self.test_a_len, self.steps)


        if self.mode_choice == 2: ##biLSTM+ mask + highway + max_pooling
            ori_q_highway, cand_a_highway = get_rnn2cnn_out(ori_q, cand_a, door_w, change_w, self.steps)
            _, neg_a_highway = get_rnn2cnn_out(ori_q, neg_a, door_w, change_w, self.steps)
            test_q_highway, test_a_highway = get_rnn2cnn_out(test_q, test_a, door_w, change_w, self.steps)
            print(ori_q_highway.shape)

            ori_q_fea, cand_a_fea= get_feature(ori_q_highway, cand_a_highway)
            ori_nq_fea, neg_a_fea = get_feature(ori_q_highway, neg_a_highway)
            test_q_fea, test_a_fea, _ = get_feature(test_q_highway, test_a_highway)

        if self.mode_choice == 3:##embedding + CNN
            ori_q_fea, cand_a_fea, neg_a_fea = cnn_qa(ori_q=ori_que, can_a=cand_que, neg_a=neg_que,
                                                      seq_len=self.steps,
                                                      hidden_size=self.embedding_size,
                                                      filter_sizes=self.filter_sizes,
                                                      num_filters=self.num_filters)
            test_q_fea, test_a_fea, _ = cnn_qa(ori_q=test_q, cand_a=test_a, neg_a=neg_que,
                                               seq_len=self.steps,
                                               hidden_size=self.embedding_size,
                                               filter_sizes=self.filter_sizes,
                                               num_filters=self.num_filters)
        if self.mode_choice == 4:## biLSTM + mask + highway + highway + maxpooling
            ori_q_highway, cand_a_highway = get_rnn2cnn_out(ori_q, cand_a, door_w, change_w, self.steps)
            _, neg_a_highway = get_rnn2cnn_out(ori_q, neg_a, door_w, change_w, self.steps)
            test_q_highway, test_a_highway = get_rnn2cnn_out(test_q, test_a, door_w, change_w, self.steps)
            ori_q_2highway, cand_a_2highway = get_rnn2cnn_out(ori_q_highway, cand_a_highway, door_w, change_w, self.steps)
            _, neg_a_2highway = get_rnn2cnn_out(ori_q_highway, neg_a_highway, door_w, change_w, self.steps)
            test_q_2highway, test_a_2highway = get_rnn2cnn_out(test_q_highway, test_a_highway, door_w, change_w, self.steps)

            ori_q_fea, cand_a_fea = get_feature(ori_q_2highway, cand_a_2highway)
            ori_nq_fea, neg_a_fea = get_feature(ori_q_2highway, neg_a_2highway)
            test_q_fea, test_a_fea, _ = get_feature(test_q_2highway, test_a_2highway)


        if self.mode_choice == 5: ## biLSTM + mask + concat + highway + attention
            ori_q_concat = tf.concat([ori_q, ori_que], 2)
            cand_a_concat = tf.concat([cand_a, cand_que], 2)
            neg_a_concat = tf.concat([neg_a, neg_que], 2)
            test_q_concat = tf.concat([test_q, test_que], 2)
            test_a_concat = tf.concat([test_a, test_ans], 2)

            print(ori_q_concat.shape)
            ori_q_highway, cand_a_highway = get_rnn2cnn_out_hxh(ori_que, ori_q_concat, cand_que, cand_a_concat, door_w, change_w, self.steps)
            print (ori_q_highway)
            _, neg_a_highway = get_rnn2cnn_out_hxh(ori_que, ori_q_concat, neg_que, neg_a_concat, door_w, change_w, self.steps)
            test_q_highway, test_a_highway = get_rnn2cnn_out(test_que, test_q_concat, test_ans, test_a_concat, door_w, change_w, self.steps)


            if self.att:
                ori_q_fea = get_feature_att(ori_q_highway, self_att, self.ori_q_len, self.steps)
                cand_a_fea = get_feature_att(cand_a_concat, self_att, self.cand_a_len, self.steps)
                neg_a_fea = get_feature_att(neg_a_concat, self_att, self.neg_a_len, self.steps)
                test_q_fea = get_feature_att(test_q_highway, self_att, self.test_q_len, self.steps)
                test_a_fea = get_feature_att(test_a_concat, self_att, self.neg_a_len, self.steps)

                self.ori_q_fea = tf.reshape(ori_q_fea, [-1,300], name='ori_q_feature')

                print ('ori_q_shape is :', ori_q_fea.shape)
            else:
                ori_q_fea, cand_a_fea = get_feature(ori_q_highway, cand_a_highway)
                _, neg_a_fea = get_feature(ori_q_highway, neg_a_highway)
                test_q_fea, test_a_fea = get_feature(test_q_highway, test_a_highway)
                self.ori_q_fea = tf.reshape(ori_q_fea, [-1, 300], name='ori_q_feature')

