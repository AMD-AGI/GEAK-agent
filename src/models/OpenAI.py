from typing import List
import openai
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential

from models.Base import BaseModel


class StandardOpenAIModel(BaseModel):
    """Standard OpenAI API (api.openai.com)"""
    def __init__(self, 
                 model_id="gpt-4o", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.client = openai.OpenAI(api_key=api_key)
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=5000) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            n=1,
            stream=False,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("No response choices returned from the API.")
        return response.choices[0].message.content


class OpenAIModel(BaseModel):
    def __init__(self, 
                 model_id="GPT4o", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id

        url = "https://llm-api.amd.com/OpenAI"
        
        try:
            user = os.getlogin()
        except Exception:
            user = os.environ.get("USER", "unknown_user")

        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "user": user
        }

        self.client = openai.OpenAI(
            base_url=url,
            api_key="dummy",
            default_headers=headers
        )
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=5000) -> str:
        # AMD API has a max_tokens limit of ~16000
        max_tokens = min(max_tokens, 16000)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            n=1,
            stream=False,
            stop=None,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=None,
            user=None
        )
        # import pdb
        # pdb.set_trace()
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("No response choices returned from the API.")

        return response.choices[0].message.content
