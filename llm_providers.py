"""
LLM Provider Interfaces
Handles connections to Ollama and LM Studio with reasoning model support
"""

import requests
import json
import re
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the LLM server is accessible"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models"""
        pass
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Generate a response from the LLM"""
        pass
    
    @staticmethod
    def strip_thinking_tags(content: str) -> str:
        """Strip reasoning/thinking content from model output"""
        if not content:
            return content
        
        thinking_patterns = [
            (r'<think>', r'</think>'),
            (r'<thinking>', r'</thinking>'),
            (r'<reason>', r'</reason>'),
            (r'<reasoning>', r'</reasoning>'),
            (r'<thought>', r'</thought>'),
            (r'<internal>', r'</internal>'),
            (r'<scratch>', r'</scratch>'),
        ]
        
        cleaned_content = content
        
        for open_tag, close_tag in thinking_patterns:
            pattern = f'{open_tag}.*?{close_tag}'
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE | re.DOTALL)
        
        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content


class OllamaProvider(LLMProvider):
    """Ollama API provider with image and reasoning model support"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = None):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
    
    def test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get('models', []):
                    name = model.get('name', '')
                    if name:
                        models.append(name)
                return sorted(models)
            return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def generate_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Generate response using Ollama chat API with image support"""
        
        if not self.model_name:
            logger.error("No model name specified")
            return None
        
        try:
            ollama_messages = []
            
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content')
                
                if isinstance(content, dict):
                    text = content.get('text', '')
                    images = content.get('images', [])
                    
                    message_data = {
                        "role": role,
                        "content": text
                    }
                    
                    if images:
                        image_list = []
                        for img in images:
                            image_data = img.get('data', '')
                            image_list.append(image_data)
                        message_data["images"] = image_list
                    
                    ollama_messages.append(message_data)
                else:
                    ollama_messages.append({
                        "role": role,
                        "content": content or ""
                    })
            
            payload = {
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": False
            }
            
            if hasattr(self, 'context_tokens'):
                payload["options"] = {"num_ctx": self.context_tokens}
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('message', {}).get('content', '')
                content = self.strip_thinking_tags(content)
                return content
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
        
        except requests.exceptions.Timeout:
            logger.error("Request timed out - model may be too slow or not loaded")
            return None
        except Exception as e:
            logger.error(f"Failed to generate Ollama response: {e}")
            return None
    
    def set_context_tokens(self, tokens: int):
        """Set the context window size in tokens"""
        self.context_tokens = tokens


class LMStudioProvider(LLMProvider):
    """LM Studio API provider with image and reasoning model support"""
    
    def __init__(self, base_url: str = "http://localhost:1234", model_name: str = None):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
    
    def test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to LM Studio: {e}")
            return False
    
    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get('data', []):
                    model_id = model.get('id', '')
                    if model_id:
                        clean_name = model_id.replace('.gguf', '')
                        parts = clean_name.split('/')
                        if len(parts) > 1:
                            clean_name = '/'.join(parts[-2:])
                        models.append(clean_name)
                return sorted(models)
            return []
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {e}")
            return []
    
    def generate_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Generate response using LM Studio's OpenAI-compatible API with image support"""
        
        try:
            openai_messages = []
            
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content')
                
                if isinstance(content, dict):
                    text = content.get('text', '')
                    images = content.get('images', [])
                    
                    if images:
                        content_parts = [{"type": "text", "text": text}]
                        
                        for img in images:
                            image_data = img.get('data', '')
                            mime_type = img.get('mime_type', 'image/jpeg')
                            
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            })
                        
                        openai_messages.append({
                            "role": role,
                            "content": content_parts
                        })
                    else:
                        openai_messages.append({
                            "role": role,
                            "content": text
                        })
                else:
                    openai_messages.append({
                        "role": role,
                        "content": content or ""
                    })
            
            payload = {
                "model": self.model_name or "local-model",
                "messages": openai_messages,
                "temperature": 0.7,
                "max_tokens": 2000,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=90
            )
            
            if response.status_code == 200:
                data = response.json()
                choices = data.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    reasoning_content = message.get('reasoning_content')
                    content = message.get('content', '')
                    
                    if not reasoning_content:
                        content = self.strip_thinking_tags(content)
                    
                    return content
                return None
            else:
                logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return None
        
        except requests.exceptions.Timeout:
            logger.error("Request timed out - model may be too slow or not loaded")
            return None
        except Exception as e:
            logger.error(f"Failed to generate LM Studio response: {e}")
            return None


class TestProvider(LLMProvider):
    """Test provider for development and debugging"""
    
    def __init__(self):
        self.counter = 0
    
    def test_connection(self) -> bool:
        return True
    
    def list_models(self) -> List[str]:
        return ["test-model-small", "test-model-large", "test-model-fast", "test-model-vision"]
    
    def generate_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        self.counter += 1
        
        last_message = messages[-1] if messages else {"content": "Hello"}
        content = last_message.get('content', '')
        
        if isinstance(content, dict):
            text = content.get('text', 'No text')
            images = content.get('images', [])
            if images:
                return f"Test response #{self.counter}: I see {len(images)} image(s) with text: '{text[:50]}...'"
        else:
            return f"Test response #{self.counter} to: '{str(content)[:50]}...'"
