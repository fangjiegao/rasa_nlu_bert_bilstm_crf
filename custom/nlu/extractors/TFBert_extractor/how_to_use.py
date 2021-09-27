# coding=utf-8
import tensorflow as tf
import os
import base_model
from transformers import BertTokenizer, BertConfig

model_name = 'bert-base-chinese'
MODEL_PATH = '../../../../data/bert-base-chinese-checkpoint/chinese_L-12_H-768_A-12'  # cd /Users/sherry/.cache/torch/
MODEL_PATH = '../../../../data/bert-checkpoint/chinese_L-12_H-768_A-12'  # cd /Users/sherry/.cache/torch/
MODEL_PATH = os.path.abspath(MODEL_PATH)  # cd /Users/sherry/.cache/torch/
VOCAB_PATH = os.path.join(MODEL_PATH, "vocab.txt")
CONFIG_PATH = os.path.join(MODEL_PATH, "config.json")

# a.通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH)
# b. 导入配置文件
model_config = BertConfig.from_pretrained(CONFIG_PATH)
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
bert_sequence_model = base_model.TFBertBasesModel.from_pretrained(
    MODEL_PATH, config=model_config, cache_dir=MODEL_PATH)

# encode仅返回input_ids
print(tokenizer.encode('我不喜欢你'))
"""
encode_plus返回所有编码信息
'input_ids':是单词在词典中的编码
'token_type_ids':区分两个句子的编码（上句全为0，下句全为1）
'attention_mask':指定对哪些词进行self-Attention操作
"""
sen_code = tokenizer.encode_plus('我不喜欢这世界', '我只喜欢你')
print(sen_code)
sen_code = tokenizer.encode_plus('我不喜欢这世界', '我只喜欢你', max_length=100, pad_to_max_length=True)
print(sen_code)

print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))

input_ids = tf.constant(tokenizer.encode("我不喜欢这世界", add_special_tokens=True))[None, :]  # Batch size 1
print(input_ids)
outputs = bert_sequence_model(input_ids, training=True)
scores = outputs[0]
print(scores.shape)
