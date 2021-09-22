# coding=utf-8
import requests

context = "男子15万造飞碟的最新相关信息"
query_info = {"inputData": [{"fileName": "illool@163.com", "text": context}]}
headers = {'content-type': 'charset=utf8'}
r = requests.post("http://localhost:5007/webhooks/bert_ner_channel", json=query_info, headers=headers)
res_json = r.content.decode('unicode_escape')
print(res_json)
