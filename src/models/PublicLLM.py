"""
Public LLM API Model - Direct support for Claude and OpenAI public APIs
No monkey-patching required!
"""

from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
from models.Base import BaseModel


class PublicLLMModel(BaseModel):
    """Unified model for public Claude and OpenAI APIs"""
    
    # Model-specific max token limits
    MODEL_MAX_TOKENS = {
        # OpenAI GPT models
        'gpt-4-turbo': 32768,
        'gpt-4': 32768,
        'gpt-4-32k': 32768,
        'gpt-3.5-turbo': 32768,
        'gpt-3.5-turbo-16k': 32768,
        'gpt-4o': 32768,
        'gpt-4o-mini': 32768,
        'gpt-5': 32768,
        'gpt-5-pro': 32768,
        # OpenAI o1/o3 reasoning models
        'o1': 100000,
        'o1-preview': 100000,
        'o1-mini': 100000,
        'o1-pro': 100000,
        'o3': 100000,
        'o3-mini': 100000,
        'o3-pro': 100000,
        # Claude models
        'claude-sonnet-4-5': 16384,  # Claude 4.5 Sonnet - 16K output tokens
        'claude-sonnet-4-5-20250514': 16384,
        'claude-opus-4': 32768,  # Claude 4 Opus
        'claude-3-7-sonnet': 32768,  # Claude 3.7 Sonnet
        'claude-3-5-sonnet-20241022': 8192,
        'claude-3-5-sonnet-20240620': 8192,
        'claude-3-5-haiku-20241022': 8192,
        'claude-3-opus-20240229': 4096,
        'claude-3-sonnet-20240229': 4096,
        'claude-3-haiku-20240307': 4096,
    }

    def __init__(self, model_id=None, api_key=None, provider=None):
        """
        Initialize public LLM model

        Args:
            model_id: Model identifier (e.g., 'claude-3-5-sonnet-20241022', 'gpt-4-turbo')
            api_key: API key for the provider
            provider: 'claude' or 'openai' (auto-detected if not specified)
        """
        assert api_key is not None, "API key is required"
        assert model_id is not None, "Model ID is required"

        self.model_id = model_id
        self.api_key = api_key

        # Auto-detect provider from model_id if not specified
        if provider is None:
            if 'claude' in model_id.lower():
                provider = 'claude'
            elif 'gpt' in model_id.lower() or 'o1' in model_id.lower() or 'o3' in model_id.lower():
                provider = 'openai'
            else:
                raise ValueError(f"Cannot auto-detect provider from model_id: {model_id}")

        self.provider = provider.lower()
        
        # Check if this is an o1/o3 model (uses different API)
        self.is_reasoning_model = any(x in model_id.lower() for x in ['o1-', 'o1mini', 'o3-', 'o3mini', 'o1 ', 'o3 ']) or model_id.lower() in ['o1', 'o3']

        # Initialize the appropriate client
        if self.provider == 'claude':
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        elif self.provider == 'openai':
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self,
                 messages: List,
                 temperature=1.0,
                 max_tokens=4096,
                 **kwargs) -> str:
        """
        Generate completion using public API

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text content
        """
        # Cap max_tokens to model's limit
        model_max = self.MODEL_MAX_TOKENS.get(self.model_id, 4096)
        if max_tokens > model_max:
            print(f"⚠️  Requested {max_tokens} tokens, but {self.model_id} supports max {model_max}. Capping to {model_max}.")
            max_tokens = model_max
        
        try:
            if self.provider == 'claude':
                # Separate system messages from conversation
                system_content = ""
                conversation_messages = []

                for msg in messages:
                    if msg['role'] == 'system':
                        system_content += msg['content'] + "\n"
                    else:
                        conversation_messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })

                # Call Claude API with streaming for large token requests
                # Streaming is required for operations that may take longer than 10 minutes
                with self.client.messages.stream(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_content.strip() if system_content else "",
                    messages=conversation_messages
                ) as stream:
                    # Collect the streamed response
                    response_text = ""
                    for text in stream.text_stream:
                        response_text += text

                return response_text

            elif self.provider == 'openai':
                # Check if this is an o1/o3 reasoning model
                if self.is_reasoning_model:
                    # o1/o3 models use the responses API endpoint
                    # Combine all messages into a single input string
                    # Note: These models don't support temperature or max_tokens parameters
                    combined_input = ""
                    for msg in messages:
                        role = msg['role']
                        content = msg['content']
                        if role == 'system':
                            combined_input += f"System: {content}\n\n"
                        elif role == 'user':
                            combined_input += f"User: {content}\n\n"
                        elif role == 'assistant':
                            combined_input += f"Assistant: {content}\n\n"
                    
                    response = self.client.responses.create(
                        model=self.model_id,
                        input=combined_input.strip()
                    )
                    
                    # Extract text from response output
                    # The output contains message items with content
                    for item in response.output:
                        if hasattr(item, 'type') and item.type == 'message':
                            if hasattr(item, 'content') and len(item.content) > 0:
                                return item.content[0].text
                    
                    # Fallback if format is unexpected
                    return str(response.output)
                else:
                    # Standard GPT models use chat completions API
                    # gpt-5 and newer models use max_completion_tokens instead of max_tokens
                    # and only support temperature=1.0
                    if 'gpt-5' in self.model_id.lower() or 'gpt-6' in self.model_id.lower():
                        # GPT-5 only supports temperature=1.0
                        if temperature != 1.0:
                            print(f"⚠️  GPT-5 only supports temperature=1.0, ignoring temperature={temperature}")
                        response = self.client.chat.completions.create(
                            model=self.model_id,
                            messages=messages,
                            max_completion_tokens=max_tokens
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model_id,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling {self.provider} API: {e}")
            raise
