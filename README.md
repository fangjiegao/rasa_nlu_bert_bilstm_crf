## 环境
rasa 2.7.2

python3.6.5

requirements.txt

## 功能
基于rasa的实现使用bert特征构建bilstm+crf模型实现序列标注任务

为了使用离线特征修改部分源码

# 训练
rasa train nlu  -c configs/config_entity.yml -u data/entity/example -vv  # 预先安装rasa

rasa_bert_bilstm_crf/rasa/__main__.py train nlu  -c configs/config_entity.yml -u data/entity/example -vv

rasa train nlu  -c configs/config_entity_v1.yml -u data/entity/example -vv  # 预先安装rasa

rasa_bert_bilstm_crf/rasa/__main__.py train nlu  -c configs/config_entity_v1.yml -u data/entity/example -vv

# 启动服务
rasa run -m models --log-file log.log -vv --port 5007 --enable-api --credentials configs/channel.yml

rasa_bert_bilstm_crf/rasa/__main__.py run -m models --log-file log.log -vv --port 5007 --enable-api --credentials configs/channel.yml

# 测试服务
python custom/client.py

# 说明
<br>bilstm+crf有两种实现方式</br>
<br>1、基于tensorflow 1.x</br>
<br>2、基于tensorflow 2.x</br>

# 结论
tensorflow 1.x的代码和rasa框架无法兼容，可赢正常训练和启动服务，但在调取服务后会出现重大的bug无法解决。
具体原因参考：

tf.compat.v1.enable_v2_behavior()
和
tf.compat.v1.disable_v2_behavior()
的用法

# 怎加基于tf2的bert语言模型的fine tune版本
## 训练
rasa train nlu  -c configs/config_entity_TF.yml -u data/entity/example -vv  # 预先安装rasa

rasa_bert_bilstm_crf/rasa/__main__.py train nlu  -c configs/config_entity_TF.yml -u data/entity/example -vv

## 启动服务
rasa run -m models --log-file log.log -vv --port 5007 --enable-api --credentials configs/channel.yml

rasa_bert_bilstm_crf/rasa/__main__.py run -m models --log-file log.log -vv --port 5007 --enable-api --credentials configs/channel.yml

##测试服务
python custom/client.py