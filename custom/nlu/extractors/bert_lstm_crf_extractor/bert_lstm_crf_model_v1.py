# coding=utf-8
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.featurizers.featurizer import Featurizer
from typing import Any, Dict, List, Optional, Text, Tuple, Type
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component
import logging
import numpy as np
from rasa.nlu.model import Metadata
from custom.nlu.extractors.bert_lstm_crf_extractor.base_model_v1 import BiLstmCrfForBertModel_v1, ORTHER_TAG

logger = logging.getLogger(__name__)
MAX_BERT_LEN = 512
BERT_VECTOR_SIZE = 768


class BiLstmCrfForBert_v1(EntityExtractor):
    name = 'bert_bilstm_crf'

    requires = [Featurizer]
    provides = ["entities"]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super().__init__(component_config)
        self.label2tag = component_config.get("label2tag", None)

    def train(self, training_data: TrainingData,
              config: Optional[RasaNLUModelConfig] = None,
              **kwargs: Any, ) -> None:
        # self.model = BiLstmCrfForBertModel(self.component_config)
        # self.label2tag = self.model.build_graph()
        self.model, self.label2tag = BiLstmCrfForBertModel_v1.instance(self.component_config)
        if training_data.entity_examples:
            filtered_entity_examples = self.filter_trainable_entities(
                training_data.training_examples
            )
            dataset = self._create_dataset(filtered_entity_examples)
            self.model.train(dataset)

    def process(self, message: Message, **kwargs: Any) -> None:
        # if self.model is None:
        self.model = BiLstmCrfForBertModel_v1(self.component_config)
        # self.model.build_graph()
        self.label2tag = self.component_config.get("label2tag")
        dataset = self._create_dataset_for_process(message, False)
        # label_id_list, seq_len_list = self.model.predict(dataset)
        label_id_list, seq_len_list = self.model.predict_by_restore_model_ckpt(dataset)

        text_list = [data[2] for data in dataset]

        entities_list = self.praser_predict_res(label_id_list, text_list)
        extracted = self.add_extractor_name(entities_list)
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)
        logger.info(self.name + "function process finish......")

    def praser_predict_res(self, label_id_list: List[List[int]], text_list: List[Text]) -> List[Dict[Text, Any]]:
        """
        return {
            "text": value, "start": _start,"end": _end,"value": value,"entity": dimensions[entity] if entity in dimensions else entity,
        }
        """
        entities_list = []
        tag_id2pos_list = self.get_entity_pos(label_id_list, text_list)
        for index in range(len(tag_id2pos_list)):
            for id_key, start_end in tag_id2pos_list[index].items():
                entities_dict = self.tag_id2tag_and_pos2text(id_key, start_end, text_list[index])
                entities_list.append(entities_dict)
        return entities_list

    def tag_id2tag_and_pos2text(self, tag_id: int, start_end: Tuple[int, int], text: Text) -> Dict[Text, Any]:
        entity_label = self.label2tag.get(str(tag_id), None)
        if entity_label is None:
            logger.error("self.label2tag no key : " + str(tag_id))
        entity_text = text[start_end[0]:start_end[1]]

        return {
            "text": entity_text,
            "start": start_end[0],
            "end": start_end[1],
            "value": entity_label.split(":")[1],
            "entity": entity_label.split(":")[0]
        }

    @classmethod
    def get_entity_pos(cls, label_id_list: List[List[int]], text_list: List[str]) -> List[Dict[int, Tuple[int, int]]]:
        if len(label_id_list) != len(text_list):
            logger.info("label_id_list", text_list)
            logger.info("text_list", text_list)
        res_list = []
        start_id = -1
        present_tag_id = -1
        for predict_res_index in range(len(label_id_list)):
            res = dict()
            if len(label_id_list[predict_res_index]) != len(text_list[predict_res_index]):
                logger.info("label_id_list", label_id_list[predict_res_index])
                logger.info("text_list", text_list[predict_res_index])
                continue
            for index in range(len(label_id_list[predict_res_index])):
                if present_tag_id == -1:
                    present_tag_id = label_id_list[predict_res_index][index]
                    start_id = index
                elif present_tag_id == label_id_list[predict_res_index][index]:
                    continue
                elif present_tag_id != label_id_list[predict_res_index][index]:
                    res[present_tag_id] = (start_id, index)
                    present_tag_id = -1
                    start_id = -1
                else:
                    pass
            res_list.append(res)
        return res_list

    @classmethod
    def load(cls, meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional["BiLstmCrfForBert_v1"] = None,
             **kwargs: Any
             ) -> "BiLstmCrfForBert_v1":

        persist_obi = cls(meta)

        return persist_obi

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        self.component_config["dropout_keep_prob"] = 1.0
        self.component_config["label2tag"] = self.label2tag
        return self.component_config

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Packages needed to be installed."""
        return [Featurizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    @classmethod
    def create(cls, component_config: Dict[Text, Any],
               config: RasaNLUModelConfig) -> 'BiLstmCrfForBert':
        return cls(component_config)

    def _create_dataset(self,
                        examples: List[Message],
                        istarin=True) -> List[Tuple[np.ndarray, List[Text], Text]]:
        dataset = []
        for example in examples:
            features_array = example.features[0].features
            if istarin is True:
                labels = self.get_label_sequence(example.data.get('entities'), len(example.data.get('text')))
            else:
                labels = []
            text = example.data['text']
            dataset.append((features_array, labels, text))
        return dataset

    def _create_dataset_for_process(self,
                                    example: Message,
                                    istarin=True) -> List[Tuple[np.ndarray, List[Text], Text]]:
        dataset = []
        # features_array = example.features[0].features
        if istarin is True:
            labels = self.get_label_sequence(example.data.get('entities'), len(example.data.get('text')))
        else:
            labels = []
        text = example.data['text']
        dataset.append((example.features[0].features, labels, text))

        return dataset

    @classmethod
    def get_label_sequence(cls, entities: List[Dict], size: int) -> List[str]:
        label_sequence = [ORTHER_TAG] * size
        for _ in entities:
            start_index = _['start']
            end_index = _['end']
            tag = _['entity'] + ":" + _['value']
            while start_index < end_index:
                label_sequence[start_index] = tag
                start_index = 1 + start_index

        return label_sequence
