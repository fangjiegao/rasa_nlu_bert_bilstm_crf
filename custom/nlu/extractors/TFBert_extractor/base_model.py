# coding=utf-8
from transformers import TFBertForTokenClassification, BertConfig, modeling_tf_bert
from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type
import os
import tensorflow as tf
import numpy as np
import threading
from custom.nlu.extractors.TFBert_extractor.utils import read_data_for_label
from custom.nlu.extractors.TFBert_extractor.crf import CRF
import shutil
import logging

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


ORTHER_TAG = "O"
MAX_LEN = 512

logger = logging.getLogger(__name__)


class TFBertBasesModel(TFBertForTokenClassification):
    _instance_lock = threading.Lock()

    def __init__(self, config, *inputs, **kwargs):
        super(TFBertBasesModel, self).__init__(config, *inputs, **kwargs)
        self.model_name = "TFBertBasesModel"
        pro_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        self.model_dir = os.path.join(pro_dir, self.model_name + "_model_path")
        self.model_path = os.path.join(self.model_dir, self.model_name + "_model_path" + ".h5")

        # Add bidirectional LSTMs
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.lstm_one = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            len(self.tag2label), return_sequences=True, activation="tanh"), merge_mode='sum')
        self.lstm_tow = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            len(self.tag2label), return_sequences=True, activation="softmax"), merge_mode='sum')
        # Add crf
        self.crf = CRF(len(self.tag2label), name='crf_layer')
        # how compile
        # self.compile(self.optimizer, loss={'crf_layer': crf.get_loss})

    def call(self, inputs, **kwargs):
        # inputs_text, attention_mask, labels = inputs
        # input_ids = inputs.get("input_ids")
        # input_shape = modeling_tf_bert.shape_list(input_ids)
        # attention_mask = tf.fill(input_shape, 1)

        outputs = self.bert(inputs, **kwargs)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output, training=kwargs.get("training", False))
        lstm_one = self.lstm_one(sequence_output)
        lstm_tow = self.lstm_tow(lstm_one)
        crf = self.crf(lstm_tow)

        return crf

    def train(self, training_data):
        labels_ids = []
        token_ids = []
        attention_mask = []
        self.sequenceLengths = []
        for (sent_seqs, tag_, text) in training_data:
            # sentence_code_list.append(sent_seqs)
            token_ids.append(sent_seqs['input_ids'])
            attention_mask.append(sent_seqs['attention_mask'])
            labels_ids.append([self.tag2label[tag] for tag in tag_])
            self.sequenceLengths.append(len(text))

        token_ids = np.asarray(token_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int32)
        labels_ids = np.array([np.array(_) for _ in labels_ids])

        sentence_code_list = [token_ids, attention_mask, labels_ids]

        class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=1000):
                super(CustomSchedule, self).__init__()

                self.d_model = tf.cast(d_model, tf.float32)
                self.warmup_steps = warmup_steps

            def __call__(self, step):
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)

                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        learning_rate = CustomSchedule(768)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # optimizer = tf.keras.optimizers.Adam(lr=1e-2)
        # self.compile(self.optimizer, loss={'crf_layer': crf.get_loss})
        self.compile(optimizer=optimizer, loss=self.crf.get_loss)

        # text_list = [temp_ids, temp_mask, labels]
        self.fit(
            x=sentence_code_list,
            y=labels_ids,
            batch_size=16,
            epochs=self.epoch_num,
            validation_data=(sentence_code_list, labels_ids)
        )

        self.init_model_dir()
        self.save_weights(self.model_path)

    @classmethod
    def instance(cls, *args, **kwargs):
        with TFBertBasesModel._instance_lock:
            if not hasattr(TFBertBasesModel, "_instance"):
                cls.data_dir = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
                model_weigths = args[0].get("model_weigths", "data/bert-checkpoint/chinese_L-12_H-768_A-12")
                cls.VOCAB_PATH = os.path.join(os.path.join(cls.data_dir, model_weigths), "vocab.txt")
                cls.CONFIG_PATH = os.path.join(os.path.join(cls.data_dir, model_weigths), "config.json")
                cls.data_file_dir = args[0].get("data_dir", "data/entity/example")
                cls.tag2label = read_data_for_label(os.path.join(cls.data_dir, cls.data_file_dir))
                cls.epoch_num = args[0].get("epoch_num", 10)
                # 导入配置文件
                model_config = BertConfig.from_pretrained(cls.CONFIG_PATH)
                # 修改配置
                model_config.output_hidden_states = True
                model_config.output_attentions = True
                model_config.num_labels = len(cls.tag2label)

                TFBertBasesModel._instance = TFBertBasesModel(model_config, **kwargs)

                # 这种方式固定了 input 的 shape
                for_call_data = {"input_ids": tf.constant([[7]*MAX_LEN])}

                if os.path.exists(TFBertBasesModel._instance.model_path):
                    # TFBertBasesModel._instance(TFBertBasesModel._instance.dummy_inputs, training=False)
                    TFBertBasesModel._instance(for_call_data, training=False)   # 这个training感觉没啥用
                    # TFBertBasesModel._instance(for_call_data, training=True)
                    # TFBertBasesModel._instance.build([None, None, len(labelid2tag)])  # doesn't work
                    TFBertBasesModel._instance.load_weights(TFBertBasesModel._instance.model_path)
                    # TFBertBasesModel._instance = tf.keras.models.load_model(TFBertBasesModel._instance.model_path)
                    logger.info("predict load model......")

            labelid2tag = TFBertBasesModel._instance.get_labelid2tag()

        return TFBertBasesModel._instance, labelid2tag

    def get_labelid2tag(self):
        return {value: key for key, value in self.tag2label.items()}

    def init_model_dir(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        else:
            shutil.rmtree(self.model_dir)
            os.mkdir(self.model_dir)

    def predict_for_TF(self, training_data: List):
        labels_ids = []
        token_ids = []
        attention_mask = []
        self.sequenceLengths = []
        for (sent_seqs, tag_, text) in training_data:
            # sentence_code_list.append(sent_seqs)
            token_ids.append(sent_seqs['input_ids'])
            attention_mask.append(sent_seqs['attention_mask'])
            labels_ids.append([self.tag2label[tag] for tag in tag_])
            self.sequenceLengths.append(len(text))

        token_ids = np.asarray(token_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int32)
        labels_ids = np.array([np.array(_) for _ in labels_ids])

        sentence_code_list = [token_ids, attention_mask, labels_ids]

        # [temp_ids, temp_mask, labels]

        label_ids = self.predict(sentence_code_list)
        return label_ids
