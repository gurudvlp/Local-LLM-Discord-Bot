"""
LLM Provider Interfaces
Handles connections to Ollama, LM Studio, Anthropic Claude, and OpenAI Codex
"""

import requests
import json
import re
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

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

    @abstractmethod
    def generate_response_stream(self, messages: List[Dict[str, Any]]):
        """Generate a response from the LLM and stream chunks"""
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

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = None, num_threads: int = None):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.num_threads = num_threads
    
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

            # Add options for context tokens and/or thread count
            options = {}
            if hasattr(self, 'context_tokens') and self.context_tokens:
                options["num_ctx"] = self.context_tokens
            if self.num_threads:
                options["num_thread"] = self.num_threads
            if options:
                payload["options"] = options
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=180
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

    def generate_response_stream(self, messages: List[Dict[str, Any]]):
        """Generate response using Ollama chat API with streaming"""

        if not self.model_name:
            logger.error("No model name specified")
            return

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
                "stream": True
            }

            # Add options for context tokens and/or thread count
            options = {}
            if hasattr(self, 'context_tokens') and self.context_tokens:
                options["num_ctx"] = self.context_tokens
            if self.num_threads:
                options["num_thread"] = self.num_threads
            if options:
                payload["options"] = options

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=180,
                stream=True
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get('message', {}).get('content', '')
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")

        except requests.exceptions.Timeout:
            logger.error("Request timed out - model may be too slow or not loaded")
        except Exception as e:
            logger.error(f"Failed to generate Ollama streaming response: {e}")


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
                timeout=180
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

    def generate_response_stream(self, messages: List[Dict[str, Any]]):
        """Generate response using LM Studio's OpenAI-compatible API with streaming"""

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
                "stream": True
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=180,
                stream=True
            )

            if response.status_code == 200:
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                        if line_str.startswith('data:'):
                            line_str = line_str[5:].strip()

                        if not line_str:
                            continue

                        try:
                            data = json.loads(line_str)
                            choices = data.get('choices', [])
                            if choices:
                                delta = choices[0].get('delta', {})
                                chunk = delta.get('content', '')
                                if chunk:
                                    yield chunk
                        except json.JSONDecodeError:
                            continue
            else:
                logger.error(f"LM Studio API error: {response.status_code} - {response.text}")

        except requests.exceptions.Timeout:
            logger.error("Request timed out - model may be too slow or not loaded")
        except Exception as e:
            logger.error(f"Failed to generate LM Studio streaming response: {e}")


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

    def generate_response_stream(self, messages: List[Dict[str, Any]]):
        """Generate test response with streaming"""
        self.counter += 1

        last_message = messages[-1] if messages else {"content": "Hello"}
        content = last_message.get('content', '')

        if isinstance(content, dict):
            text = content.get('text', 'No text')
            images = content.get('images', [])
            if images:
                response_text = f"Test response #{self.counter}: I see {len(images)} image(s) with text: '{text[:50]}...'"
            else:
                response_text = f"Test response #{self.counter} (dict): {text[:50]}..."
        else:
            response_text = f"Test response #{self.counter} to: '{str(content)[:50]}...'"

        # Simulate streaming by yielding words
        words = response_text.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider with vision and streaming support"""

    PREDEFINED_MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5"
    ]

    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-5"):
        self.api_key = api_key
        self.model_name = model_name

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.anthropic = anthropic
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic>=0.40.0")
            raise ImportError("anthropic package not installed")

    def test_connection(self) -> bool:
        """Test API key with minimal request"""
        try:
            # Make a minimal API call to verify the key
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except self.anthropic.AuthenticationError:
            logger.error("Claude authentication failed - invalid API key")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Claude API: {e}")
            return False

    def list_models(self) -> List[str]:
        """Return predefined list of Claude models"""
        return self.PREDEFINED_MODELS

    def _convert_messages_to_claude_format(self, messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict]]:
        """
        Convert bot's internal message format to Claude's API format.

        Returns:
            (system_prompt, claude_messages)
        """
        claude_messages = []
        system_message = None

        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')

            # Extract system message separately (Claude expects it as a parameter)
            if role == 'system':
                system_message = content
                continue

            # Handle text-only messages
            if isinstance(content, str):
                claude_messages.append({
                    "role": role,
                    "content": content
                })
                continue

            # Handle multimodal messages (text + images)
            if isinstance(content, dict):
                content_blocks = []

                # Add text block
                if content.get('text'):
                    content_blocks.append({
                        "type": "text",
                        "text": content['text']
                    })

                # Add image blocks
                for img in content.get('images', []):
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.get('mime_type', 'image/jpeg'),
                            "data": img['data']
                        }
                    })

                claude_messages.append({
                    "role": role,
                    "content": content_blocks
                })

        return system_message, claude_messages

    def generate_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Generate response using Claude API"""
        try:
            system_prompt, claude_messages = self._convert_messages_to_claude_format(messages)

            # Build request parameters
            params = {
                "model": self.model_name,
                "max_tokens": 2000,
                "messages": claude_messages
            }

            if system_prompt:
                params["system"] = system_prompt

            # Make API call
            response = self.client.messages.create(**params)

            # Extract text from response
            if response.content:
                text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
                return ' '.join(text_blocks)

            return None

        except self.anthropic.RateLimitError as e:
            logger.error(f"Claude rate limit: {e}")
            return "⏳ API rate limit reached. Please try again in a moment."

        except self.anthropic.AuthenticationError as e:
            logger.error(f"Claude authentication failed: {e}")
            return "❌ API authentication failed. Please check your API key."

        except self.anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return f"❌ API error occurred: {str(e)}"

        except Exception as e:
            logger.error(f"Unexpected error in Claude provider: {e}")
            return None

    def generate_response_stream(self, messages: List[Dict[str, Any]]):
        """Generate response using Claude API with streaming"""
        try:
            system_prompt, claude_messages = self._convert_messages_to_claude_format(messages)

            # Build request parameters
            params = {
                "model": self.model_name,
                "max_tokens": 2000,
                "messages": claude_messages
            }

            if system_prompt:
                params["system"] = system_prompt

            # Make streaming API call
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    yield text

        except self.anthropic.RateLimitError as e:
            logger.error(f"Claude rate limit: {e}")
            yield "⏳ API rate limit reached. Please try again in a moment."

        except self.anthropic.AuthenticationError as e:
            logger.error(f"Claude authentication failed: {e}")
            yield "❌ API authentication failed. Please check your API key."

        except self.anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            yield f"❌ API error occurred: {str(e)}"

        except Exception as e:
            logger.error(f"Unexpected error in Claude streaming: {e}")


class CodexOAuthManager:
    """Handles OAuth 2.0 flow for OpenAI Codex authentication"""

    # OAuth endpoints for OpenAI
    AUTH_URL = "https://auth.openai.com/authorize"
    TOKEN_URL = "https://auth.openai.com/oauth/token"
    CLIENT_ID = "pdlliX90HHCGW3wkH16PuIhV"  # OpenAI CLI client ID
    REDIRECT_URI = "http://localhost:8090/auth/callback"

    TOKENS_DIR = Path(".tokens")

    def __init__(self):
        self.TOKENS_DIR.mkdir(exist_ok=True)

    def start_oauth_flow(self) -> Optional[Dict[str, str]]:
        """
        Start OAuth flow and return tokens.

        Process:
        1. Generate authorization URL
        2. User opens browser and signs in
        3. Browser redirects to localhost with code
        4. User pastes the redirect URL
        5. Extract code and exchange for tokens
        """
        import secrets

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)

        # Build authorization URL
        auth_params = {
            "client_id": self.CLIENT_ID,
            "response_type": "code",
            "redirect_uri": self.REDIRECT_URI,
            "scope": "openai",
            "state": state
        }

        auth_url = f"{self.AUTH_URL}?" + "&".join(f"{k}={v}" for k, v in auth_params.items())

        print("\n" + "=" * 70)
        print("OpenAI Codex OAuth Authentication")
        print("=" * 70)
        print("\n1. Open this URL in your browser:")
        print(f"\n   {auth_url}\n")
        print("2. Sign in with your ChatGPT Plus account")
        print("3. After signing in, the browser will redirect to localhost")
        print("   (You'll see an error page - that's expected!)")
        print("4. Copy the FULL URL from your browser's address bar")
        print("   (It should start with: http://localhost:8090/auth/callback?code=...)")
        print("\n" + "=" * 70 + "\n")

        # Wait for user to paste the redirect URL
        redirect_url = input("Paste the full redirect URL here: ").strip()

        if not redirect_url:
            logger.error("No redirect URL provided")
            return None

        # Extract code from URL
        try:
            parsed_url = urlparse(redirect_url)
            params = parse_qs(parsed_url.query)

            code = params.get('code', [None])[0]
            returned_state = params.get('state', [None])[0]

            if not code:
                logger.error("No authorization code found in URL")
                return None

            # Verify state matches (CSRF protection)
            if returned_state != state:
                logger.warning("State mismatch - possible CSRF attack")

            # Exchange code for tokens
            tokens = self.exchange_code_for_tokens(code)
            return tokens

        except Exception as e:
            logger.error(f"Failed to parse redirect URL: {e}")
            return None

    def exchange_code_for_tokens(self, code: str) -> Optional[Dict[str, str]]:
        """Exchange authorization code for access and refresh tokens"""
        try:
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.REDIRECT_URI,
                "client_id": self.CLIENT_ID
            }

            response = requests.post(
                self.TOKEN_URL,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )

            if response.status_code == 200:
                tokens = response.json()
                # Add expiry timestamp
                if 'expires_in' in tokens:
                    expiry = datetime.now() + timedelta(seconds=tokens['expires_in'])
                    tokens['expires_at'] = expiry.isoformat()

                return tokens
            else:
                logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {e}")
            return None

    def refresh_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Get new access token using refresh token"""
        try:
            token_data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.CLIENT_ID
            }

            response = requests.post(
                self.TOKEN_URL,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )

            if response.status_code == 200:
                tokens = response.json()
                # Add expiry timestamp
                if 'expires_in' in tokens:
                    expiry = datetime.now() + timedelta(seconds=tokens['expires_in'])
                    tokens['expires_at'] = expiry.isoformat()

                return tokens
            else:
                logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return None

    def save_tokens(self, tokens: Dict[str, str], config_name: str) -> bool:
        """Save tokens to file"""
        try:
            token_file = self.TOKENS_DIR / f"{config_name}_codex_tokens.json"
            with open(token_file, 'w') as f:
                json.dump(tokens, f, indent=2)
            logger.info(f"Saved tokens to {token_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
            return False

    def load_tokens(self, config_name: str) -> Optional[Dict[str, str]]:
        """Load tokens from file"""
        try:
            token_file = self.TOKENS_DIR / f"{config_name}_codex_tokens.json"
            if not token_file.exists():
                logger.warning(f"Token file not found: {token_file}")
                return None

            with open(token_file, 'r') as f:
                tokens = json.load(f)
            return tokens
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            return None


class OpenAICodexProvider(LLMProvider):
    """OpenAI Codex provider with OAuth authentication"""

    PREDEFINED_MODELS = [
        "gpt-4o",
        "o1",
        "o1-mini"
    ]

    def __init__(self, access_token: str = None, refresh_token: str = None,
                 model_name: str = "gpt-4o", config_name: str = None):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.model_name = model_name
        self.config_name = config_name
        self.token_expiry = None
        self.oauth_manager = CodexOAuthManager()

        try:
            import openai
            self.openai = openai
            if access_token:
                self.client = openai.OpenAI(api_key=access_token)
            else:
                self.client = None
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai>=1.0.0")
            raise ImportError("openai package not installed")

        # Load token expiry if available
        if config_name:
            tokens = self.oauth_manager.load_tokens(config_name)
            if tokens and 'expires_at' in tokens:
                try:
                    self.token_expiry = datetime.fromisoformat(tokens['expires_at'])
                except:
                    pass

    def is_token_expired(self) -> bool:
        """Check if access token is expired or expiring soon"""
        if not self.token_expiry:
            # If we don't know expiry, assume it might be expired
            return True

        # Refresh if expiring within 5 minutes
        buffer_time = timedelta(minutes=5)
        return datetime.now() + buffer_time >= self.token_expiry

    def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False

        logger.info("Refreshing OAuth access token...")
        new_tokens = self.oauth_manager.refresh_token(self.refresh_token)

        if not new_tokens:
            logger.error("Failed to refresh token")
            return False

        # Update tokens
        self.access_token = new_tokens.get('access_token', self.access_token)
        if 'refresh_token' in new_tokens:
            self.refresh_token = new_tokens['refresh_token']

        # Update expiry
        if 'expires_at' in new_tokens:
            try:
                self.token_expiry = datetime.fromisoformat(new_tokens['expires_at'])
            except:
                pass

        # Update client
        self.client = self.openai.OpenAI(api_key=self.access_token)

        # Save updated tokens
        if self.config_name:
            self.oauth_manager.save_tokens(new_tokens, self.config_name)

        logger.info("Access token refreshed successfully")
        return True

    def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token, refreshing if needed"""
        if self.is_token_expired():
            return self.refresh_access_token()
        return True

    def test_connection(self) -> bool:
        """Test API connection with minimal request"""
        try:
            if not self.client:
                return False

            # Ensure token is valid
            if not self._ensure_valid_token():
                return False

            # Make a minimal API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return True
        except self.openai.AuthenticationError:
            logger.error("OpenAI authentication failed - invalid or expired token")
            # Try to refresh
            if self.refresh_access_token():
                return self.test_connection()  # Retry once
            return False
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI API: {e}")
            return False

    def list_models(self) -> List[str]:
        """Return predefined list of Codex models"""
        return self.PREDEFINED_MODELS

    def _convert_messages_to_openai_format(self, messages: List[Dict[str, Any]]) -> List[Dict]:
        """
        Convert bot's internal message format to OpenAI's API format.
        Reuses logic from LMStudioProvider since it's OpenAI-compatible.
        """
        openai_messages = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content')

            if isinstance(content, dict):
                text = content.get('text', '')
                images = content.get('images', [])

                if images:
                    # Multimodal content
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
                    # Text only
                    openai_messages.append({
                        "role": role,
                        "content": text
                    })
            else:
                # Simple text message
                openai_messages.append({
                    "role": role,
                    "content": content or ""
                })

        return openai_messages

    def generate_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Generate response using OpenAI API"""
        try:
            # Ensure token is valid
            if not self._ensure_valid_token():
                return "❌ Failed to refresh authentication token. Please re-authenticate."

            openai_messages = self._convert_messages_to_openai_format(messages)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=2000
            )

            if response.choices:
                return response.choices[0].message.content

            return None

        except self.openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit: {e}")
            return "⏳ API rate limit reached. Please try again in a moment."

        except self.openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            # Try to refresh token
            if self.refresh_access_token():
                return self.generate_response(messages)  # Retry once
            return "❌ API authentication failed. Please re-authenticate."

        except self.openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return f"❌ API error occurred: {str(e)}"

        except Exception as e:
            logger.error(f"Unexpected error in OpenAI provider: {e}")
            return None

    def generate_response_stream(self, messages: List[Dict[str, Any]]):
        """Generate response using OpenAI API with streaming"""
        try:
            # Ensure token is valid
            if not self._ensure_valid_token():
                yield "❌ Failed to refresh authentication token. Please re-authenticate."
                return

            openai_messages = self._convert_messages_to_openai_format(messages)

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=2000,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except self.openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit: {e}")
            yield "⏳ API rate limit reached. Please try again in a moment."

        except self.openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            # Try to refresh token
            if self.refresh_access_token():
                # Retry streaming
                for chunk in self.generate_response_stream(messages):
                    yield chunk
            else:
                yield "❌ API authentication failed. Please re-authenticate."

        except self.openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            yield f"❌ API error occurred: {str(e)}"

        except Exception as e:
            logger.error(f"Unexpected error in OpenAI streaming: {e}")
