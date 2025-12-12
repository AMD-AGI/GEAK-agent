from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
from google import genai
from google.genai.types import HttpOptions
from models.Base import BaseModel


class GeminiModel(BaseModel):
    def __init__(self, 
                 model_id="gemini-2.5-pro-preview-05-06", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        
        try:
            user = os.getlogin()
        except OSError:
            user = os.environ.get("USER", "unknown_user")

        self.client = genai.Client(
            vertexai=True,
            api_key="dummy",
            http_options=HttpOptions(
                base_url="https://llm-api.amd.com/VertexGen",
                api_version="v1",
                headers={
                    "Ocp-Apim-Subscription-Key": api_key,
                    "user": user
                }
            )
        )

    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=1.0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=30000) -> str:
        
        # Convert messages list to a single string prompt
        prompt = ""
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                prompt += str(msg['content']) + "\n"
            else:
                prompt += str(msg) + "\n"
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt.strip(),
            config={
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                'top_p': 0.95,
                'presence_penalty': presence_penalty,
                'frequency_penalty': frequency_penalty,
            }
        )
        
        return response.text