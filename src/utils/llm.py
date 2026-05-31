"""
Azure OpenAI interface for the Agentic Deep Research System.
"""
import json
from typing import List, Optional, Dict, Any, Union
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

from .config import config, AzureOpenAIConfig


class LLMClient:
    """Azure OpenAI client wrapper with retry logic."""
    
    def __init__(self, azure_config: Optional[AzureOpenAIConfig] = None):
        self.config = azure_config or config.azure_openai
        
        self.client = AzureOpenAI(
            azure_endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            api_version=self.config.api_version
        )
        
        self.embedding_client = AzureOpenAI(
            azure_endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            api_version=self.config.embedding_api_version,
            timeout=120.0,
            max_retries=5
        )
        
        # Token counter
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """Send chat completion request."""
        params = {
            "model": self.config.chat_deployment,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            params["response_format"] = response_format
        
        params.update(kwargs)
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Send chat completion request expecting JSON response."""
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Failed to parse JSON response: {response}")
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """Embed a single batch with retry logic."""
        response = self.embedding_client.embeddings.create(
            model=self.config.embedding_deployment,
            input=batch
        )
        return [item.embedding for item in response.data]

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Batch embeddings (max 16 at a time for Azure)
        all_embeddings = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


# Global LLM client instance
llm_client = LLMClient()


def chat(messages: List[Dict[str, str]], **kwargs) -> str:
    """Convenience function for chat completion."""
    return llm_client.chat(messages, **kwargs)


def chat_json(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Convenience function for JSON chat completion."""
    return llm_client.chat_json(messages, **kwargs)


def embed(texts: Union[str, List[str]]) -> List[List[float]]:
    """Convenience function for embeddings."""
    return llm_client.embed(texts)
