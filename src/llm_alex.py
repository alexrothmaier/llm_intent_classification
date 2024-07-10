import torch

from pydantic import BaseModel

from typing import List, Optional, Any, Union

from transformers import pipeline, Pipeline

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# from src.ml.classifier.llm.util.hf_login import login_to_hf

# from src.ml.classifier.llm.api.base import BaseRemoteLLM

# login_to_hf()


class Llama(LLM):

    _name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    _system_msg: str = "You are a helpful assistant"
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(_device)
    pipeline: Optional[Pipeline] = None
    terminators: List[Union[None, int]] = []
    #cache: dict = dict()

    class Config:
        arbitrary_types_allowed=True

    def setup(self):
        self.pipeline = pipeline(
            "text-generation",
            model=self._name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=self._device,
            pad_token_id=128009
        )

        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]




    def __call__(self, *, query: str, **kwargs) -> str:

        prompt = query
        if self.pipeline is None:
            self.setup()

        if self.pipeline is None:
            raise ValueError("Pipeline is not initialized")

        if hasattr(self.pipeline, "tokenizer") is False:
            raise ValueError("Tokenizer is not initialized")

        messages = [
            {"role": "system", "content": self._system_msg},
            {"role": "user", "content": prompt},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )

        #if prompt in self.cache:
            #return self.cache[prompt]



        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.001,
            top_p=0.9
        )
        input_tokens_length = len(self.pipeline.tokenizer.encode(prompt))
        prompt_length = len(prompt)
        
        answer = outputs[0]["generated_text"][len(prompt):]

        try:
            # extract json_str from the answer
            # Extract the JSON part from the text
            #start_index = answer.find('{')
            #end_index = answer.rfind('}') + 1
            #answer = answer[start_index:end_index]
            answer = answer
            return answer
        except Exception as e:
            print(e)
       # self.cache[prompt] = answer
        return  {
            "answer": answer,
            "input_tokens": input_tokens_length,
            "prompt_length": prompt_length
        }
        

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:

       return self.__call__(query=prompt)
 
    @property
    def _llm_type(self) -> str:
        return self.name



# 
