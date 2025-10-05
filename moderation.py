"""
Content Moderation Service
Integrates with OpenAI's free moderation API
"""

import aiohttp
import asyncio
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModerationService:
    """Handles content moderation using OpenAI's free moderation API"""
    
    MODERATION_URL = "https://api.openai.com/v1/moderations"
    
    THRESHOLDS = {
        "harassment": 0.7,
        "harassment/threatening": 0.7,
        "hate": 0.7,
        "hate/threatening": 0.7,
        "self-harm": 0.7,
        "self-harm/intent": 0.7,
        "self-harm/instructions": 0.7,
        "sexual": 0.7,
        "sexual/minors": 0.3,  # Lower threshold for child safety
        "violence": 0.7,
        "violence/graphic": 0.7
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.enabled = bool(api_key)
        
        if not self.enabled:
            logger.warning("Moderation service initialized without API key - moderation disabled")
    
    async def check_content(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if content is safe
        Returns: (is_safe, reason_if_not_safe)
        """
        
        if not self.enabled:
            return True, None
        
        if not text or not text.strip():
            return True, None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": text[:2000]  # API limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.MODERATION_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._process_moderation_result(data)
                    elif response.status == 401:
                        logger.error("Invalid OpenAI API key for moderation")
                        self.enabled = False
                        return True, None  # Fail open
                    else:
                        logger.error(f"Moderation API error: {response.status}")
                        return True, None
        
        except asyncio.TimeoutError:
            logger.warning("Moderation API timeout - allowing content")
            return True, None
        except Exception as e:
            logger.error(f"Moderation check failed: {e}")
            return True, None  # Fail open on error
    
    def _process_moderation_result(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Process the moderation API response"""
        
        try:
            results = data.get('results', [])
            if not results:
                return True, None
            
            result = results[0]
            
            if result.get('flagged', False):
                categories = result.get('categories', {})
                triggered = [cat for cat, flagged in categories.items() if flagged]
                
                if triggered:
                    reason = self._format_violation_reason(triggered)
                    return False, reason
            
            # Check scores against custom thresholds
            scores = result.get('category_scores', {})
            violations = []
            
            for category, score in scores.items():
                threshold = self.THRESHOLDS.get(category, 0.8)
                if score > threshold:
                    violations.append(category)
            
            if violations:
                reason = self._format_violation_reason(violations)
                return False, reason
            
            return True, None
        
        except Exception as e:
            logger.error(f"Error processing moderation result: {e}")
            return True, None
    
    def _format_violation_reason(self, categories: list) -> str:
        """Format violation categories into human-readable reason"""
        
        category_descriptions = {
            "harassment": "harassment",
            "harassment/threatening": "threatening behavior",
            "hate": "hate speech",
            "hate/threatening": "threatening hate speech",
            "self-harm": "self-harm content",
            "self-harm/intent": "self-harm intent",
            "self-harm/instructions": "self-harm instructions",
            "sexual": "sexual content",
            "sexual/minors": "content involving minors",
            "violence": "violence",
            "violence/graphic": "graphic violence"
        }
        
        violations = []
        for cat in categories:
            desc = category_descriptions.get(cat, cat.replace('/', ' '))
            if desc not in violations:
                violations.append(desc)
        
        if len(violations) == 1:
            return f"Contains {violations[0]}"
        elif len(violations) == 2:
            return f"Contains {violations[0]} and {violations[1]}"
        else:
            return f"Contains multiple policy violations including {violations[0]}"
    
    async def test_connection(self) -> bool:
        """Test if the moderation API is accessible with the current key"""
        
        if not self.enabled:
            return False
        
        try:
            is_safe, _ = await self.check_content("Hello, this is a test message.")
            return is_safe is not None
        except Exception as e:
            logger.error(f"Moderation API test failed: {e}")
            return False


class ModerationFilter:
    """Additional local moderation filters"""
    
    @staticmethod
    def check_spam(text: str, history: list = None) -> bool:
        """Check for spam patterns"""
        
        # Check for repetition in message history
        if history and len(history) > 2:
            recent_messages = [msg.get('content', '') for msg in history[-3:]]
            if all(msg == text for msg in recent_messages):
                return False
        
        # Check for excessive caps
        if len(text) > 10:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.8:
                return False
        
        # Check for excessive repetition within message
        words = text.lower().split()
        if len(words) > 5:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                return False
        
        return True
    
    @staticmethod
    def check_mention_spam(text: str) -> bool:
        """Check for excessive mentions"""
        
        mention_count = text.count('<@') + text.count('@everyone') + text.count('@here')
        return mention_count < 5
