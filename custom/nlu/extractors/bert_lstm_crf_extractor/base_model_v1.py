# coding=utf-8
import os
import shutil
import time
from custom.nlu.extractors.bert_lstm_crf_extractor.utils import batch_yield, read_dictionary, read_data_for_label, \
    conlleval, pad_sequences, pad_labels
from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type
import tensorflow as tf
import threading
import tensorflow_addons as tfa
import logging


MAX_BERT_LEN = 512
ORTHER_TAG = "O"
BERT_VECTOR_SIZE = 768
logger = logging.getLogger(__name__)


class BiLstmCrfForBertModel_v1(object):
    _instance_lock = threading.Lock()

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        self.name = "BiLstmCrfForBertModel"
        self.batch_size = component_config.get("batch_size", 64)
        self.epoch_num = component_config.get("epoch_num", 10)
        self.hidden_dim = component_config.get("hidden_dim", 256)
        self.CRF = component_config.get("CRF", True)
        self.dropout_keep_prob = component_config.get("dropout_keep_prob", 0.1)
        self.optimizer = component_config.get("optimizer", "Adam")
        self.lr = component_config.get("lr", 0.001)
        self.clip = component_config.get("clip", 5.0)
        self.shuffle = component_config.get("shuffle", True)

        self.vocab_path = component_config.get("vocab_path", "data/bert-checkpoint/vocab.txt")

        vocab_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        self.vocab = read_dictionary(os.path.join(vocab_dir, self.vocab_path))

        self.data_dir = component_config.get("data_dir", "data/entity/example")
        data_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

        self.tag2label = read_data_for_label(os.path.join(data_dir, self.data_dir))
        self.num_tags = len(self.tag2label)

        self.model_dir = os.path.join(vocab_dir, self.name + "_model_path")
        self.model_path = os.path.join(self.model_dir, self.name + "_model_path")
        self.summary_path = os.path.join(vocab_dir, self.name + "_summary_path")
        self.result_path = os.path.join(vocab_dir, self.name + "_result_path")

        # Session configuration
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory
        self.config = config

    @classmethod
    def instance(cls, *args, **kwargs):
        with BiLstmCrfForBertModel_v1._instance_lock:
            if not hasattr(BiLstmCrfForBertModel_v1, "_instance"):
                BiLstmCrfForBertModel_v1._instance = BiLstmCrfForBertModel_v1(*args, **kwargs)
            label2tag = BiLstmCrfForBertModel_v1._instance.build_graph()
        return BiLstmCrfForBertModel_v1._instance, label2tag

    def init_all_path(self):
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        else:
            shutil.rmtree(self.result_path)
            os.mkdir(self.result_path)

        if not os.path.exists(self.summary_path):
            os.mkdir(self.summary_path)
        else:
            shutil.rmtree(self.summary_path)
            os.mkdir(self.summary_path)

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        else:
            shutil.rmtree(self.model_dir)
            os.mkdir(self.model_dir)

    def init_model_dir(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        else:
            shutil.rmtree(self.model_dir)
            os.mkdir(self.model_dir)

    def build_graph(self):
        self.add_placeholders()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()
        return {value: key for key, value in self.tag2label.items()}

    def add_placeholders(self):
        # tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()
        self.word_embeddings = tf.compat.v1.placeholder(tf.float32, shape=[None, None, 768], name="word_embeddings")
        self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.compat.v1.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="lr")

    def biLSTM_layer_op(self):
        with tf.compat.v1.variable_scope("bi-lstm"):
            cell_fw = tf.compat.v1.nn.rnn_cell.LSTMCell(self.hidden_dim)
            cell_bw = tf.compat.v1.nn.rnn_cell.LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.compat.v1.variable_scope("proj"):
            w = tf.compat.v1.get_variable(name="w",
                                          shape=[2 * self.hidden_dim, self.num_tags],
                                          initializer=tf.random_normal_initializer(),
                                          dtype=tf.float32)

            b = tf.compat.v1.get_variable(name="b",
                                          shape=[self.num_tags],
                                          initializer=tf.zeros_initializer(),
                                          dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            # pred = tf.matmul(output, w) + b
            pred = tf.add(tf.matmul(output, w), b, name='op_pred')

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags], name="op_logits_gfj")

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32, name="op_softmax")

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(inputs=self.logits,
                                                                                 tag_indices=self.labels,
                                                                                 sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood, name="op_loss")

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def trainstep_op(self):
        with tf.compat.v1.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip, self.clip), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.compat.v1.global_variables_initializer()

    def add_summary(self, sess):
        """
        :param sess:
        :return:
        """
        self.merged = tf.compat.v1.summary.merge_all()
        self.file_writer = tf.compat.v1.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, training_data: List) -> None:
        self.init_all_path()
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        with tf.compat.v1.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, training_data, training_data, epoch)
            # save model
            self.init_model_dir()
            saver.save(sess, self.model_path)

    def predict(self, test_data: List) -> Tuple[List, List]:
        saver = tf.compat.v1.train.Saver()

        with tf.compat.v1.Session(config=self.config) as sess:
            logger.info(self.name + ':predict')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test_data)
            # self.evaluate(label_list, seq_len_list, test_data)
            return label_list, seq_len_list

    def run_one_epoch(self, sess, train, dev, epoch):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels, text) in enumerate(batches):

            logger.info(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, text, labels, self.lr, self.dropout_keep_prob)
            logits, _, loss_train, step_num_ = sess.run([self.logits, self.train_op, self.loss, self.global_step],
                                                        feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            # self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                # saver.save(sess, self.model_path, global_step=step_num)
                pass

        logger.info(self.name + ':validation')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, text, labels=None, lr=None, dropout=None):
        """
        :param seqs:
        :param labels:
        :param text:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        seq_len_list = [len(_) for _ in text]

        seqs = pad_sequences(seqs, text)

        feed_dict = {self.word_embeddings: seqs,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_labels(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        # else:
        #    feed_dict[self.labels] = [[0]*len(text_) for text_ in text]
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    @staticmethod
    def get_feed_dict_to_graph(seqs, text, word_embeddings, sequence_lengths, dropout_pl):
        seq_len_list = [len(_) for _ in text]
        seqs = pad_sequences(seqs, text)

        feed_dict = {word_embeddings: seqs, sequence_lengths: seq_len_list, dropout_pl: 0.0}

        return feed_dict, seq_len_list

    def predict_one_batch(self, sess, seqs, text):
        """
        :param sess:
        :param seqs:
        :param text:
        :return: label_list
                 seq_len_list
        """
        # feed_dict, seq_len_list = self.get_feed_dict(seqs, text, dropout=1.0)
        feed_dict, seq_len_list = self.get_feed_dict(seqs, text, dropout=0.0)
        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = tfa.text.viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag

        model_predict = []
        for label_, (_, tag, text) in zip(label_list, data):
            label_res = label_[:len(text)]
            tag_ = [label2tag[label__] for label__ in label_res]
            sent_res = []
            if len(label_res) != len(text):
                logger.info(text)
                logger.info(len(label_res))
                logger.info(tag)
                logger.info(text)
            for i in range(len(text)):
                sent_res.append([text[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch + 1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        # for _ in conlleval(model_predict, label_path, metric_path):
        #    logger.info(_)
        conlleval(model_predict, label_path, metric_path)

    def dev_one_epoch(self, sess, dev):
        """
        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels, text in batch_yield(dev, self.batch_size, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs, text)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_by_restore_model_ckpt(self, dev):
        if tf.executing_eagerly():
            tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()
        sess = tf.compat.v1.Session()
        ckpt_file_path = os.path.join(self.model_dir, self.name + "_model_path")
        logger.info(ckpt_file_path + ".meta")
        saver = tf.compat.v1.train.import_meta_graph(ckpt_file_path + ".meta")  # 加载模型结构
        saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))  # 只需要指定目录就可以恢复所有变量信息
        # 直接获取保存的变量
        # logger.info(sess.run('b:0'))

        # 获取placeholder变量
        bert_feature = sess.graph.get_tensor_by_name('word_embeddings:0')
        labels = sess.graph.get_tensor_by_name('labels:0')
        sequence_lengths = sess.graph.get_tensor_by_name('sequence_lengths:0')
        dropout_pl = sess.graph.get_tensor_by_name('dropout:0')
        # 获取需要进行计算的operator
        op_logits = sess.graph.get_tensor_by_name('proj/op_logits_gfj:0')

        label_list, seq_len_list = [], []

        if self.CRF:
            # 定义crf函数
            log_likelihood, op_transition_params = tfa.text.crf_log_likelihood(inputs=op_logits,
                                                                               tag_indices=labels,
                                                                               sequence_lengths=sequence_lengths)
            sess.run(tf.compat.v1.global_variables_initializer())

            for seqs, labels, text in batch_yield(dev, self.batch_size, self.tag2label, shuffle=False):
                feed_dict, seq_len_list_ = BiLstmCrfForBertModel_v1.get_feed_dict_to_graph(seqs, text, bert_feature,
                                                                                        sequence_lengths, dropout_pl)

                logits, transition_params = sess.run([op_logits, op_transition_params], feed_dict=feed_dict)
                label_list_ = []
                for logit, seq_len in zip(logits, seq_len_list_):
                    viterbi_seq, _ = tfa.text.viterbi_decode(logit[:seq_len], transition_params)
                    label_list_.append(viterbi_seq)
                    label_list.extend(label_list_)
                    seq_len_list.extend(seq_len_list_)
                    print(logits)
        else:
            # 获取placeholder变量
            bert_feature = sess.graph.get_tensor_by_name('word_embeddings:0')
            sequence_lengths = sess.graph.get_tensor_by_name('sequence_lengths:0')
            dropout_pl = sess.graph.get_tensor_by_name('dropout:0')
            op_softmax = sess.graph.get_tensor_by_name('op_softmax:0')

            for seqs, labels, text in batch_yield(dev, self.batch_size, self.tag2label, shuffle=False):
                feed_dict, seq_len_list_ = BiLstmCrfForBertModel_v1.get_feed_dict_to_graph(seqs, text, bert_feature,
                                                                                        sequence_lengths, dropout_pl)

                label_list_ = sess.run(op_softmax, feed_dict=feed_dict)
                label_list.extend(label_list_)
                seq_len_list.extend(seq_len_list_)
        sess.close()

        if not tf.executing_eagerly():
            # Enable tensorflow Eager execution
            # tf.compat.v1.enable_v2_behavior()
            pass
        sess.close()
        return label_list, seq_len_list
