# coding=utf-8
from rasa.nlu.extractors.extractor import EntityExtractor
# from rasa.nlu.featurizers.featurizer import Featurizer
from typing import Any, Dict, List, Optional, Text, Tuple, Type
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component
import logging
import numpy as np
import os
from transformers import BertTokenizer
from rasa.nlu.model import Metadata
from custom.nlu.extractors.TFBert_extractor.utils import get_max_len, pad_sequences_one
from custom.nlu.extractors.TFBert_extractor.base_model import TFBertBasesModel, ORTHER_TAG, MAX_LEN
from rasa.shared.nlu.constants import TEXT, ENTITY_ATTRIBUTE_START, ENTITY_ATTRIBUTE_END, ENTITY_ATTRIBUTE_VALUE, \
    ENTITY_ATTRIBUTE_TYPE

logger = logging.getLogger(__name__)


class TFBiLstmCrfForBert(EntityExtractor):
    name = 'TFbert_bilstm_crf'
    provides = ["entities"]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super().__init__(component_config)
        self.label2tag = component_config.get("label2tag", None)

    def train(self, training_data: TrainingData,
              config: Optional[RasaNLUModelConfig] = None,
              **kwargs: Any, ) -> None:

        self.model, self.label2tag = TFBertBasesModel.instance(self.component_config)

        # 通过词典导入分词器
        self.tokenizer = BertTokenizer.from_pretrained(TFBertBasesModel.VOCAB_PATH)

        if training_data.entity_examples:
            filtered_entity_examples = self.filter_trainable_entities(
                training_data.training_examples
            )
            dataset = self._create_dataset(filtered_entity_examples)
            self.model.train(dataset)

    def process(self, message: Message, **kwargs: Any) -> None:
        self.model, _ = TFBertBasesModel.instance(self.component_config)
        # 通过词典导入分词器
        self.tokenizer = BertTokenizer.from_pretrained(TFBertBasesModel.VOCAB_PATH)
        self.label2tag = self.component_config.get("label2tag")
        dataset = self._create_dataset_for_process(message, False)
        label_id_list = self.model.predict_for_TF(dataset)

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
            TEXT: entity_text,
            ENTITY_ATTRIBUTE_START: start_end[0],
            ENTITY_ATTRIBUTE_END: start_end[1],
            ENTITY_ATTRIBUTE_VALUE: entity_label.split(":")[1],
            ENTITY_ATTRIBUTE_TYPE: entity_label.split(":")[0]
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
            # [CLS]......[SEP]
            if len(label_id_list[predict_res_index][1:len(text_list[predict_res_index])+1]) != len(text_list[predict_res_index]):
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
             cached_component: Optional["TFBiLstmCrfForBert"] = None,
             **kwargs: Any
             ) -> "TFBiLstmCrfForBert":

        persist_obi = cls(meta)
        return persist_obi

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        self.component_config["dropout_keep_prob"] = 1.0
        self.component_config["label2tag"] = self.label2tag
        return self.component_config

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Packages needed to be installed."""
        return []

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["transformers"]

    @classmethod
    def create(cls, component_config: Dict[Text, Any],
               config: RasaNLUModelConfig) -> 'TFBiLstmCrfForBert':
        created_obi = cls(component_config)
        return created_obi

    def _create_dataset(self,
                        examples: List[Message],
                        istarin=True) -> List[Tuple[np.ndarray, List[Text], Text]]:
        dataset = []
        # maxLen = get_max_len(examples) + 2
        maxLen = MAX_LEN

        for example in examples:
            text = example.data.get("text")
            sentence_code = self.tokenizer.encode_plus(text, max_length=maxLen, pad_to_max_length=True)
            logger.info(self.tokenizer.convert_ids_to_tokens(sentence_code['input_ids']))
            if istarin is True:
                labels = self.get_label_sequence(example.data.get('entities'), maxLen)
            else:
                labels = []
            dataset.append((sentence_code, labels, text))
        return dataset

    def _create_dataset_for_process(self,
                                    example: Message,
                                    istarin=True) -> List[Tuple[np.ndarray, List[Text], Text]]:
        dataset = []
        maxLen = get_max_len([example]) + 2
        maxLen = MAX_LEN
        # features_array = example.features[0].features
        if istarin is True:
            # labels = self.get_label_sequence(example.data.get('entities'), len(example.data.get('text')))
            labels = self.get_label_sequence(example.data.get('entities'), maxLen)
        else:
            # labels = []
            labels = self.get_label_sequence(example.data.get('entities'), maxLen)
        text = example.data['text']
        sentence_code = self.tokenizer.encode_plus(text, max_length=maxLen, pad_to_max_length=True)
        dataset.append((sentence_code, labels, text))

        return dataset

    @classmethod
    def get_label_sequence(cls, entities: List[Dict], size: int) -> List[str]:
        label_sequence = [ORTHER_TAG] * size
        for _ in entities:
            start_index = _['start'] + 1  # [CLS]......[SEP]
            end_index = _['end'] + 1  # [CLS]......[SEP]
            tag = _['entity'] + ":" + _['value']
            while start_index < end_index:
                label_sequence[start_index] = tag
                start_index = 1 + start_index

        return label_sequence
