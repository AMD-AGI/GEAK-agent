# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

"""
OpenAI API interface for LLMs
"""
import os
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import openai

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        assert model_cfg.api_key or os.environ.get('OPENAI_API_KEY'), "API key must be provided either in config.yaml or as an environment variable 'OPENAI_API_KEY'"
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.reasoning_effort = model_cfg.reasoning_effort
        self.api_key = model_cfg.api_key if model_cfg.api_key else os.environ.get('OPENAI_API_KEY')
        self.random_seed = getattr(model_cfg, 'random_seed', None)
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.api_key 
        }
        # Set up API client
        self.client = openai.OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY', 'dummy'),
            base_url=self.api_base,
            default_headers=self.headers
        )

        logger.info(f"Initialized OpenAI LLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        if self.api_base == "https://api.openai.com/v1" and str(self.model).lower().startswith("o"):
            # For o-series models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
        elif "o1" in self.model.lower() or "o3" in self.model.lower():
            # For o1 and o2 models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                # "temperature": kwargs.get("temperature", self.temperature),
                "reasoning_effort": self.reasoning_effort,
                "top_p": kwargs.get("top_p", self.top_p),
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
        elif "gpt" in self.model.lower() or "o4" in self.model.lower():
            params = {
                "model": self.model,
                "messages": formatted_messages,
                # "temperature": kwargs.get("temperature", self.temperature),
                # "reasoning_effort": self.reasoning_effort,
                "top_p": kwargs.get("top_p", self.top_p),
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
        else:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
        
        # Add seed parameter for reproducibility if configured
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # print(response)
        data = response.model_dump()

        if 'gemini' in params["model"]:
            # print(f'response : {data["candidates"][0]["content"]["parts"][0]["text"]}')
            # Logging of system prompt, user message and response content
            logger = logging.getLogger(__name__)
            logger.debug(f"API parameters: {params}")
            logger.debug(f"API response: {data['candidates'][0]['content']['parts'][0]['text']}")
            return data["candidates"][0]["content"]["parts"][0]["text"] 
        elif 'claude' in params["model"]:
            # print(f'response : {data['content'][0]["text"]}')
            # Logging of system prompt, user message and response content
            logger = logging.getLogger(__name__)
            logger.debug(f"API parameters: {params}")
            logger.debug(f"API response: {data['content'][0]['text']}")
            return data['content'][0]["text"]
        elif 'o1' in params["model"].lower() or 'o3' in params["model"].lower() or "gpt" in params["model"].lower() or "o4" in params["model"].lower():
            # For o1 and o3 models, return the first choice's message content
            logger = logging.getLogger(__name__)
            logger.debug(f"API parameters: {params}")
            logger.debug(f"API response: {response.choices[0].message.content}")
            return response.choices[0].message.content

    # async def _call_api(self, params: Dict[str, Any]) -> str:
    #     """Make the actual API call"""
    #     # Use asyncio to run the blocking API call in a thread pool
    #     loop = asyncio.get_event_loop()
    #     response = await loop.run_in_executor(
    #         None, lambda: self.client.chat.completions.create(**params)
    #     )
    #     # Logging of system prompt, user message and response content
    #     logger = logging.getLogger(__name__)
    #     logger.debug(f"API parameters: {params}")
    #     logger.debug(f"API response: {response.choices[0].message.content}")
    #     return response.choices[0].message.content
