# coding=utf-8
import logging
import typing
from typing import Any, Dict, List, Optional, Text

from rasa.nlu.components import Component
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class BertTokenizer(Tokenizer):

    supported_language_list = ["zh"]
    provides = ["char_tokens"]
    defaults = {}

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new intent classifier using the MITIE framework."""

        super().__init__(component_config)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return []

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:

        text = message.get(attribute)

        tokenized = self.get_chinese_char(text)
        tokens = [Token(word, start) for (word, start, end) in tokenized]

        # maximum sequence length of 512 tokens
        if len(tokens) > 510:
            tokens = tokens[0:510]

        logger.info(str([_.text for _ in tokens]))

        return self._apply_token_pattern(tokens)

    @staticmethod
    def get_chinese_char(text: str):
        index = 0
        for ch in text:
            yield ch, index, index+1
            index = index + 1

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional[Component] = None,
        **kwargs: Any,
    ) -> "BertTokenizer":
        return cls()

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        pass
