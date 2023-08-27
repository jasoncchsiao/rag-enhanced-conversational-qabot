from transformers import pipeline
from transformers import AutoTokenizer
from typing import Dict, List
from sagemaker.predictor import Predictor
from sagemaker.deserializer import JSONDeserializer
from sagemaker.serializer import JSONSerializer
import time
import traceback

MAX_TOKENS = 1500
LLM_ENDPOINT = 'huggingface-pytorch-tgi-inference'


class Reader:
    def __init__(self, original_prompt: str, summarizer_model: str=None, summarizer_tokenizer: str=None,
                 llm_tokenizer: str=None, memory_on: bool=True) -> None:
        self.memory_on = memory_on
        self.messages = [{"role": "System", "content": original_prompt},]
        self.prompt = original_prompt
        self.summarizer = pipeline(
            'summarization',
            model=summarizer_model,
            tokenizer=summarizer_tokenizer,
            max_length=200
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer)
        self.payload = {
            "inputs": self.prompt,
            "parameters": {
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 0.2,
                "max_new_tokens": 500,
                "repetition_penalty": 1.9,
                "stop": ["\nUser:","\nUser","\nAssistant","\nAssistant:"]
            }
        }
        self.llm = Predictor(
            endpoint_name=LLM_ENDPOINT,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )

    def get_summarizer(self, message_from_llm: str) -> str:
        return self.summarizer(message_from_llm)[0]['summary_text']

    def query_llm(self, user_input: str, relevant_context: str) -> str:
        llm_response = 'I cannot find the answer in the text.'
        try:
            llm_response = self._get_response(user_input, relevant_context)
        except:
            traceback.print_exc()
            llm_response = "Sorry, messages are too long."
        return llm_response

    def _update_user_input(self, user_input: str, relevant_context: str) -> str:
        return relevant_context + user_input

    def _get_response(self, user_input: str, relevant_context: str) -> str:
        user_input = self._update_user_input(user_input, relevant_context)
        if self.memory_on:
            self.messages.append({"role": "User", "content": user_input})
            self.prompt += user_input
        else:
            # RESET self.messages everytime IMPORTANT!
            self.messages = [{"role": "User", "content": user_input}]
            # RESET self.prompt as well IMPORTANT!
            self.prompt = user_input
        self.payload["inputs"] = self.prompt
        print("PAYLOAD to feed LLM=", "\n"+self.payload["inputs"])
        response = self.llm.predict(self.payload)
        assist_msg = response[0]["generated_text"]
        if assist_msg.endswith("User:") or assist_msg.endswith("User"):
            assist_msg = response[0]["generated_text"][len(self.prompt): -len("User:")]
        elif assist_msg.endswith("Assistant:") or assist_msg.endswith("Assistant"):
            assist_msg = response[0]["generated_text"][len(self.prompt): -len("Asssistant:")]
        else:
            assist_msg = response[0]["generated_text"][len(self.prompt):]
        self.prompt += assist_msg
        self.payload["inputs"] = self.prompt
        while len(self.messages) > 2 and self.count_tokens():
            self.messages = self.messages[:1] + self.messages[2:]
        if self.count_tokens() > MAX_TOKENS:
            self.messages = self.messages[:1]
            assist_msg = "Sorry, messages are too long."
        else:
            self.messages.append({"role": "Assistant", "content": assist_msg})
        return assist_msg

    def count_tokens(self) -> int:
        num_tokens = 0
        for message in self.messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += self.tokenizer.tokenize(value).__len__()
                num_tokens += 3
        return num_tokens

if __name__ == '__main__':
    original_prompt = "You are a helpful Assistant."
    qabot = Reader(original_prompt,
                   "../../notebooks/my_saved_models/t5-small",
                   "../../notebooks/my_saved_models/t5-small/tokenizer",
                   "tiiuae/falcon-7b-instruct")


