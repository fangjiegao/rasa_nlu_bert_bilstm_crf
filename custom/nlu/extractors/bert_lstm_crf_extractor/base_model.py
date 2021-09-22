import tensorflow as tf
from custom.nlu.extractors.bert_lstm_crf_extractor.utils import read_data_for_label
from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from custom.nlu.extractors.bert_lstm_crf_extractor.crf import CRF
import threading
import numpy as np
import shutil
import logging
# from tensorflow_addons.layers import CRF

MAX_BERT_LEN = 512
ORTHER_TAG = "O"
BERT_VECTOR_SIZE = 768

logger = logging.getLogger(__name__)


class BiLstmCrfForBertModel:
    _instance_lock = threading.Lock()

    def __init__(self, component_config, vecSize=BERT_VECTOR_SIZE, maxLen=MAX_BERT_LEN):
        self.model = None
        self.name = "BiLstmCrfForBertModel"
        self.vecSize = vecSize
        self.maxLen = maxLen
        self.data_dir = component_config.get("data_dir", "data/entity/example")
        self.epoch_num = component_config.get("epoch_num", 1000)
        self.optimizer = component_config.get("optimizer", "adam")
        data_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        self.tag2label = read_data_for_label(os.path.join(data_dir, self.data_dir))
        self.num_tags = len(self.tag2label)

        pro_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        self.log = os.path.join(pro_dir, self.name + "_log")

        self.model_dir = os.path.join(pro_dir, self.name + "_model_path")
        self.model_path = os.path.join(self.model_dir, self.name + "_model_path" + ".h5")

    @classmethod
    def instance(cls, *args, **kwargs):
        with BiLstmCrfForBertModel._instance_lock:
            if not hasattr(BiLstmCrfForBertModel, "_instance"):
                BiLstmCrfForBertModel._instance = BiLstmCrfForBertModel(*args, **kwargs)
            label2tag = BiLstmCrfForBertModel._instance.buildBiLSTMCRF()

            if os.path.exists(BiLstmCrfForBertModel._instance.model_path):
                BiLstmCrfForBertModel._instance.model.load_weights(BiLstmCrfForBertModel._instance.model_path)

        return BiLstmCrfForBertModel._instance, label2tag

    def getTransParam(self, label_ids):
        transParam = np.zeros([len(list(self.tag2label.keys())), len(list(self.tag2label.keys()))])
        for rowI in range(len(label_ids)):

            for colI in range(len(label_ids[rowI]) - 1):
                transParam[label_ids[rowI][colI]][label_ids[rowI][colI + 1]] += 1
        for rowI in range(transParam.shape[0]):
            transParam[rowI] = transParam[rowI] / np.sum(transParam[rowI])
        return transParam

    def buildBiLSTMCRF(self):
        self.model = Sequential()
        self.model.add(tf.keras.layers.Input(shape=(None, self.vecSize)))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.num_tags, return_sequences=True, activation="tanh"), merge_mode='sum'))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.num_tags, return_sequences=True, activation="softmax"), merge_mode='sum'))
        crf = CRF(self.num_tags, name='crf_layer')
        self.model.add(crf)
        self.model.compile(self.optimizer, loss={'crf_layer': crf.get_loss})
        return {value: key for key, value in self.tag2label.items()}

    def init_model_dir(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        else:
            shutil.rmtree(self.model_dir)
            os.mkdir(self.model_dir)

    def init_log_dir(self):
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        else:
            shutil.rmtree(self.log)
            os.mkdir(self.log)

    def train(self, training_data: List):
        word_embeddings = []
        labels_ids = []
        self.sequenceLengths = []
        for (sent_seqs, tag_, text) in training_data:
            word_embeddings.append(sent_seqs)
            labels_ids.append([self.tag2label[tag] for tag in tag_])
            self.sequenceLengths.append(len(text))

        # self.transParam = self.getTransParam(labels_ids)
        labels_ids = [np.array(_) for _ in labels_ids]

        self.init_log_dir()
        tensorboard_callback = TensorBoard(log_dir=self.log, histogram_freq=1)
        history = self.model.fit(np.array(word_embeddings), np.array(labels_ids), epochs=self.epoch_num, callbacks=[tensorboard_callback])

        self.init_model_dir()
        self.model.save_weights(self.model_path)
        return history

    def predict(self, training_data: List):
        word_embeddings = []
        for (sent_seqs, _, _) in training_data:
            word_embeddings.append(sent_seqs)

        if self.model is None:
            self.label2tag = self.buildBiLSTMCRF()
            self.model.load_weights(self.model_path)
            logger.info("predict load model......")

        label_ids = self.model.predict(np.array(word_embeddings))

        return label_ids
