# coding=utf-8
import inspect
import logging
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from typing import Text, Optional, Callable, Awaitable, Any, Dict, List

from rasa.core.channels.channel import (
    InputChannel,
    UserMessage,
)

logger = logging.getLogger(__name__)


class BertNerInput(InputChannel):

    @classmethod
    def name(cls) -> Text:
        return "bert_ner_channel"

    # noinspection PyMethodMayBeStatic
    def _extract_input_data(self, req: Request) -> Optional[List]:
        return req.json.get("inputData", None)

    # noinspection PyMethodMayBeStatic
    def _extract_user_id(self, req: Request) -> Optional[Text]:
        return req.json.get("userId", None)

    # noinspection PyMethodMayBeStatic
    def validate_inner_data(self, data: Dict):
        if any(map(lambda i: i not in data, ['fileName', 'text'])):
            raise KeyError(f'fileName or text not in Input Data')
        else:
            return str(data['fileName']), str(data['text'])

    @staticmethod
    def format_input_data(input_data: List[Dict]):
        return [(str(i['fileName']), str(i['text'])) for i in input_data]

    def blueprint(self,
                  on_new_message: Callable[[UserMessage], Awaitable[Any]]) -> Blueprint:
        custom_webhook = Blueprint(
            "custom_webhook_{}".format(type(self).__name__),
            inspect.getmodule(self).__name__,
        )

        @custom_webhook.route("/", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:
            input_datas = self._extract_input_data(request)

            # noinspection PyBroadException
            try:
                datas = self.format_input_data(input_datas)
                output_channel = Text2gpOutput(datas)

                for _name, text in datas:
                    data = await on_new_message(
                        UserMessage(
                            text,
                            None,
                        )
                    )
                    output_channel.add_data_to_channel(data)
                return output_channel.send_text_message()
            except KeyError as e:
                logger.error(f'inputData error ! inputData: {input_datas}')
                return response.json({'statusCode': 250,
                                      'message': str(e),
                                      'result': '',
                                      'recommend': ''})

            except Exception as e:
                logger.exception(
                    f"An exception occur while handling "
                    f"user message '{input_datas}'."
                )
                return response.json({'statusCode': 250,
                                      'message': str(e),
                                      'result': '',
                                      'recommend': ''})

        return custom_webhook


class Text2gpOutput:

    def __init__(self, input_data: List):
        self.status = 200  # default
        self.message = ''
        self.result_queue = []
        self.input_data = input_data

    def send_text_message(self) -> HTTPResponse:
        template = {"statusCode": 200,
                    "message": "调用模型内部服务接口成功",
                    "result": self.result_queue
                    }

        for idx, (_name, _) in enumerate(self.input_data):
            self.result_queue[idx]['fileName'] = _name

        return response.json(template)

    def add_data_to_channel(self, data: Dict):
        """Transform data to target format."""
        text = data.get('text', '')
        entities = data.get('entities', [])
        result = {'content': text, 'entities': entities}

        self.result_queue.append(result)
        return result
