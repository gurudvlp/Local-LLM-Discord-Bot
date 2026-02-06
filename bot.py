"""
Discord Bot Implementation
Handles Discord interactions and manages conversations
"""

import discord
from discord import app_commands
from discord.ext import commands
import asyncio
from typing import Dict, List, Optional, Any, Deque
from datetime import datetime
import logging
from collections import defaultdict, deque
import base64
import aiohttp
import io
import json
from pathlib import Path
import sys
import re

from llm_providers import OllamaProvider, LMStudioProvider, ClaudeProvider, OpenAICodexProvider, OpenAIAPIProvider, CodexOAuthManager
from moderation import ModerationService
from personality_manager import PersonalityManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLogger:
    """Handles chat logging and RLHF data collection"""
    
    def __init__(self, chat_logging_enabled: bool = False, rlhf_logging_enabled: bool = False, config_name: str = "unknown"):
        self.chat_logging_enabled = chat_logging_enabled
        self.rlhf_logging_enabled = rlhf_logging_enabled
        self.config_name = config_name
        
        if self.chat_logging_enabled:
            self.chat_logs_dir = Path("chat_logs")
            self.chat_logs_dir.mkdir(exist_ok=True)
            self.current_chat_log = self._create_chat_log_file()
        
        if self.rlhf_logging_enabled:
            self.rlhf_logs_dir = Path("rlhf_logs")
            self.rlhf_logs_dir.mkdir(exist_ok=True)
            self.current_rlhf_log = self._create_rlhf_log_file()
    
    def _create_chat_log_file(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.chat_logs_dir / f"chatlog_{self.config_name}_{timestamp}.jsonl"
    
    def _create_rlhf_log_file(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.rlhf_logs_dir / f"rlhf_{self.config_name}_{timestamp}.jsonl"
    
    def log_chat(self, user_message: str, assistant_response: str, system_prompt: Optional[str] = None):
        """Log chat exchange in ChatML format"""
        if not self.chat_logging_enabled:
            return
        
        try:
            messages = []
            
            # Only log custom system prompts
            if system_prompt and system_prompt != "You are a helpful AI assistant. Be friendly and concise.":
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_response})
            
            log_entry = {"messages": messages}
            with open(self.current_chat_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to log chat: {e}")
    
    def log_rlhf(self, user_message: str, assistant_response: str, rating: str, system_prompt: Optional[str] = None):
        """Log RLHF feedback with rating ('good' or 'bad')"""
        if not self.rlhf_logging_enabled:
            return
        
        try:
            messages = []
            
            if system_prompt and system_prompt != "You are a helpful AI assistant. Be friendly and concise.":
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_response})
            
            log_entry = {
                "messages": messages,
                "rating": rating
            }
            with open(self.current_rlhf_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to log RLHF feedback: {e}")


class ConversationManager:
    """Manages conversation context with shifting window"""
    
    def __init__(self, max_messages: int = 10, system_prompt: str = None, is_multi_user: bool = False):
        self.max_messages = max_messages
        self.system_prompt = system_prompt or "You are a helpful AI assistant. Be friendly and concise."
        self.is_multi_user = is_multi_user
        self.history: Deque[Dict[str, Any]] = deque(maxlen=max_messages)
        self.last_exchange = None
    
    def add_exchange(self, user_message: Any, assistant_response: str, username: Optional[str] = None):
        """Add user-assistant exchange to history"""
        if isinstance(user_message, str):
            raw_user_message = user_message
            # Add username prefix for multi-user contexts
            if self.is_multi_user and username:
                formatted_message = f"[{username}]: {user_message}"
            else:
                formatted_message = user_message
        else:
            # Multimodal messages with images
            raw_user_message = user_message.get('text', '') if isinstance(user_message, dict) else str(user_message)
            formatted_message = user_message
        
        self.last_exchange = {
            'user': raw_user_message,
            'assistant': assistant_response
        }
        
        self.history.append({
            'user': formatted_message,
            'assistant': assistant_response
        })
    
    def get_messages_for_llm(self, current_message: Any, username: Optional[str] = None, personalities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Build message list for LLM including system prompt, personalities, and history"""
        messages = []

        system_content = self.system_prompt

        # Add personality instructions if provided
        if personalities:
            system_content += "\n\n" + "\n".join(personalities)

        if self.is_multi_user:
            system_content += "\n\nYou are in a group chat with multiple users. Each user message will show [username]: before their message to identify who is speaking. You should respond without using a [botname]: prefix or any similar prefix for your own responses."
        elif username:
            # In single-user mode (DMs), inform the LLM of the user's name
            system_content += f"\n\nYou are chatting with a user named {username}."

        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history
        for exchange in self.history:
            user_msg = exchange['user']
            if isinstance(user_msg, dict):
                messages.append({"role": "user", "content": user_msg.get('content', user_msg)})
            else:
                messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": exchange['assistant']})
        
        # Add current message
        if isinstance(current_message, str):
            if self.is_multi_user and username:
                current_message = f"[{username}]: {current_message}"
            messages.append({"role": "user", "content": current_message})
        else:
            messages.append({"role": "user", "content": current_message})
        
        return messages
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
        self.last_exchange = None


class DiscordLLMBot:
    """Main Discord bot with LLM integration"""
    
    MAX_RLHF_CACHE_SIZE = 1000
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        intents.reactions = True
        
        self.bot = commands.Bot(
            command_prefix='!',
            intents=intents,
            description="Local LLM Discord Bot"
        )
        
        # Initialize LLM provider
        provider_type = config['llm_provider']

        if provider_type == 'ollama':
            self.llm_provider = OllamaProvider(
                base_url=config['llm_base_url'],
                model_name=config['model_name'],
                num_threads=config.get('ollama_num_threads')
            )

        elif provider_type == 'lm_studio':
            self.llm_provider = LMStudioProvider(
                base_url=config['llm_base_url'],
                model_name=config['model_name']
            )

        elif provider_type == 'claude':
            self.llm_provider = ClaudeProvider(
                api_key=config['claude_api_key'],
                model_name=config['model_name']
            )

        elif provider_type == 'openai_codex':
            # Load OAuth tokens from file
            tokens = self._load_codex_tokens(config['_config_name'])
            self.llm_provider = OpenAICodexProvider(
                access_token=tokens.get('access_token'),
                refresh_token=tokens.get('refresh_token'),
                model_name=config['model_name'],
                config_name=config['_config_name']
            )

        elif provider_type == 'openai_api':
            self.llm_provider = OpenAIAPIProvider(
                api_key=config['openai_api_key'],
                model_name=config['model_name']
            )

        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        # Initialize moderation if enabled
        self.moderation = None
        if config.get('moderation_enabled'):
            self.moderation = ModerationService(config.get('openai_api_key'))
        
        # Lowercase whitelists for case-insensitive comparison
        self.dm_whitelist = [u.lower() for u in config.get('dm_whitelist', [])]
        self.server_whitelist = [u.lower() for u in config.get('server_whitelist', [])]
        self.owner_username = config.get('owner_username', '').lower()
        
        config_name = config.get('_config_name', 'unknown')
        
        self.data_logger = DataLogger(
            chat_logging_enabled=config.get('chat_logging_enabled', False),
            rlhf_logging_enabled=config.get('rlhf_logging_enabled', False),
            config_name=config_name
        )

        self.personality_manager = PersonalityManager()

        self.conversations: Dict[int, ConversationManager] = {}
        self.processing: set = set()
        self.message_cache: Dict[int, Dict[str, Any]] = {}
        
        self._setup_events()
        self._setup_commands()

    def _load_codex_tokens(self, config_name: str) -> Dict:
        """Load OAuth tokens for OpenAI Codex"""
        oauth_manager = CodexOAuthManager()
        tokens = oauth_manager.load_tokens(config_name)
        return tokens if tokens else {}
    
    def is_user_whitelisted(self, user: discord.User, is_dm: bool) -> bool:
        username = user.name.lower()
        
        if is_dm:
            return username in self.dm_whitelist
        else:
            return username in self.server_whitelist
    
    def is_owner(self, user: discord.User) -> bool:
        return user.name.lower() == self.owner_username

    def is_channel_whitelisted(self, message: discord.Message) -> bool:
        """
        Check if the message is in a whitelisted server/channel combination.
        Returns True if channel is whitelisted, False otherwise.
        Supports both IDs and names for flexible matching.
        In whitelisted channels, any user can interact with the bot (no user whitelist check needed).
        """
        # DMs are never "whitelisted channels" - use user whitelist only
        if isinstance(message.channel, discord.DMChannel):
            return False

        whitelisted_servers = self.config.get('whitelisted_servers', [])

        # No whitelisted servers configured
        if not whitelisted_servers:
            return False

        # Get current server and channel info
        guild_id = message.guild.id
        guild_name = message.guild.name.lower()
        channel_id = message.channel.id
        channel_name = message.channel.name.lower()

        # Check each whitelisted server
        for server_config in whitelisted_servers:
            # Check if this is the right server (by ID or name)
            server_match = False

            if 'server_id' in server_config and server_config['server_id'] == guild_id:
                server_match = True
            elif 'server' in server_config and server_config['server'].lower() == guild_name:
                server_match = True

            if not server_match:
                continue

            # Server matched, now check channels
            channel_ids = server_config.get('channel_ids', [])
            channel_names = [c.lower() for c in server_config.get('channels', [])]

            # Check if current channel is whitelisted
            if channel_id in channel_ids:
                return True
            if channel_name in channel_names:
                return True

        return False

    def _resolve_system_prompt(self) -> str:
        """Resolve system prompt, handling file:// URIs"""
        raw_prompt = self.config.get('system_prompt', '')

        if not raw_prompt.startswith('file:'):
            return raw_prompt

        # Extract file path from "file:/path/to/file" format
        file_path = raw_prompt[5:]  # Remove "file:" prefix

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"System prompt file is empty: {file_path}")
                    return "You are a helpful AI assistant. Be friendly and concise."
                return content
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {file_path}")
            return "You are a helpful AI assistant. Be friendly and concise."
        except Exception as e:
            logger.error(f"Error reading system prompt file {file_path}: {e}")
            return "You are a helpful AI assistant. Be friendly and concise."

    def _get_conversation_manager(self, channel_id: int, is_dm: bool) -> ConversationManager:
        if channel_id not in self.conversations:
            self.conversations[channel_id] = ConversationManager(
                max_messages=self.config.get('context_window_size', 10),
                system_prompt=self._resolve_system_prompt(),
                is_multi_user=not is_dm
            )
        return self.conversations[channel_id]
    
    async def _download_image(self, url: str) -> Optional[bytes]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        return await response.read()
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
        return None
    
    async def _process_attachments(self, message: discord.Message) -> Optional[List[Dict[str, str]]]:
        """Convert image attachments to base64 for vision models"""
        images = []
        
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                image_data = await self._download_image(attachment.url)
                if image_data:
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    images.append({
                        'type': 'image',
                        'data': base64_image,
                        'mime_type': attachment.content_type
                    })
                    logger.info(f"Processed image attachment: {attachment.filename}")
        
        return images if images else None
    
    @staticmethod
    def strip_username_prefix(text: str) -> str:
        """Remove [username]: prefix from bot responses in multi-user mode"""
        if not text:
            return text
        
        pattern = r'^\[([^\]]+)\]:\s*'
        cleaned_text = re.sub(pattern, '', text, count=1)
        
        return cleaned_text
    
    def _setup_events(self):
        
        @self.bot.event
        async def on_ready():
            logger.info(f'Bot logged in as {self.bot.user}')
            print(f'\n✅ Bot is online as {self.bot.user}')
            
            config_name = self.config.get('_config_name', 'unknown')
            print(f'📁 Using config: {config_name}')
            
            # Detect vision model
            model_name = self.config.get('model_name', '').lower()
            if 'vision' in model_name or 'llava' in model_name or 'bakllava' in model_name or 'minicpm' in model_name:
                print(f'\n🖼️  Vision model detected! You can send images with your messages.')
            
            print(f'\n📱 Ready to chat!')
            print(f'   • In servers: mention the bot or reply to its messages')
            print(f'   • To DM: Right-click the bot in any server → "Message"')
            print(f'   • Still works with /chat command too')
            
            print(f'\n🔒 Access Control:')
            print(f'   • Owner: {self.owner_username}')
            print(f'   • DM whitelist: {len(self.dm_whitelist)} user(s)')
            print(f'   • Server whitelist: {len(self.server_whitelist)} user(s)')
            
            if self.config.get('moderation_enabled'):
                print(f'\n🛡️  Moderation is ENABLED - scanning messages for inappropriate content')
            
            if self.config.get('chat_logging_enabled') or self.config.get('rlhf_logging_enabled'):
                print(f'\n📊 Data Collection:')
                if self.config.get('chat_logging_enabled'):
                    print(f'   • Chat logging: ENABLED (owner only) → chat_logs/{config_name}_*.jsonl')
                if self.config.get('rlhf_logging_enabled'):
                    print(f'   • RLHF logging: ENABLED (react with 👍/👎) → rlhf_logs/{config_name}_*.jsonl')
            
            dm_channels = sum(1 for c in self.bot.private_channels if isinstance(c, discord.DMChannel))
            server_count = len(self.bot.guilds)
            
            if server_count > 0:
                print(f'\n📊 Active in {server_count} server(s)')
            if dm_channels > 0:
                print(f'💬 {dm_channels} DM conversation(s)')
            
            try:
                synced = await self.bot.tree.sync()
                print(f'\n✅ Commands synced!')
            except Exception as e:
                logger.error(f'Failed to sync commands: {e}')
                print(f'\n⚠️  Commands might take a few minutes to appear')
        
        @self.bot.event
        async def on_reaction_add(reaction: discord.Reaction, user: discord.User):
            """Handle RLHF feedback via reactions"""
            
            if user == self.bot.user:
                return
            
            if not self.is_owner(user):
                return
            
            if not self.data_logger.rlhf_logging_enabled:
                return
            
            if str(reaction.emoji) not in ['👍', '👎']:
                return
            
            if reaction.message.author != self.bot.user:
                return
            
            if reaction.message.id not in self.message_cache:
                return
            
            cached_data = self.message_cache[reaction.message.id]
            rating = "good" if str(reaction.emoji) == '👍' else "bad"
            
            self.data_logger.log_rlhf(
                user_message=cached_data['user_message'],
                assistant_response=cached_data['bot_response'],
                rating=rating,
                system_prompt=self._resolve_system_prompt()
            )
            
            # Prevent memory growth
            if len(self.message_cache) > self.MAX_RLHF_CACHE_SIZE:
                oldest_ids = sorted(self.message_cache.keys())[:self.MAX_RLHF_CACHE_SIZE // 2]
                for msg_id in oldest_ids:
                    del self.message_cache[msg_id]
            
            logger.info(f"RLHF feedback logged: {rating} from {user.name}")
        
        @self.bot.event
        async def on_message(message: discord.Message):
            
            if message.author == self.bot.user:
                return
            
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_whitelisted_channel = self.is_channel_whitelisted(message)

            # Allow if whitelisted channel OR whitelisted user
            if not is_whitelisted_channel:
                if not self.is_user_whitelisted(message.author, is_dm):
                    logger.info(f"User {message.author.name} not whitelisted for {'DM' if is_dm else 'server'} use")
                    return
            
            should_respond = False

            if is_dm:
                should_respond = True
            elif is_whitelisted_channel:
                # Full speech access in whitelisted channels
                should_respond = True
            else:
                # Non-whitelisted: require mention or reply
                # Check for mention
                if self.bot.user in message.mentions:
                    should_respond = True

                # Check for reply to bot
                if message.reference:
                    try:
                        replied_msg = await message.channel.fetch_message(message.reference.message_id)
                        if replied_msg.author == self.bot.user:
                            should_respond = True
                    except:
                        pass
            
            # Check if this is a prefix command (don't treat commands as regular chat)
            is_command = message.content.startswith(self.bot.command_prefix)

            if should_respond:
                await self._handle_chat_message(message, is_dm)

            await self.bot.process_commands(message)
    
    async def _handle_chat_message(self, message: discord.Message, is_dm: bool):
        # Skip if this is a prefix command (let process_commands handle it)
        if message.content.startswith(self.bot.command_prefix):
            return

        if message.channel.id in self.processing:
            await message.add_reaction('⏳')
            return

        self.processing.add(message.channel.id)
        
        try:
            async with message.channel.typing():
                # Remove bot mentions from content
                content = message.content
                for mention in message.mentions:
                    content = content.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
                content = content.strip()
                
                images = await self._process_attachments(message)
                
                if not content and not images:
                    await message.reply("You need to say something or send an image! 😊")
                    return
                
                raw_content = content
                
                # Check moderation on user input
                if self.moderation and content:
                    is_safe, reason = await self.moderation.check_content(content)
                    if not is_safe:
                        print(f"\n⚠️  MODERATION TRIGGERED for user {message.author.name}:")
                        print(f"   Reason: {reason}")
                        print(f"   Message: {content[:100]}{'...' if len(content) > 100 else ''}")
                        
                        await message.reply("⚠️ Your message was flagged by the content filter. Please keep the conversation appropriate.")
                        return
                
                conv_manager = self._get_conversation_manager(message.channel.id, is_dm)
                username = message.author.display_name  # Always capture username, even in DMs

                if images:
                    user_message = {
                        'text': content or "What's in this image?",
                        'images': images
                    }
                else:
                    user_message = content

                personalities = self.personality_manager.get_personalities(message.channel.id)
                llm_messages = conv_manager.get_messages_for_llm(user_message, username, personalities)

                # Use streaming or non-streaming based on config
                if self.config.get('enable_streaming', True):
                    stream_generator = self._generate_response_stream(llm_messages)
                    sent_message = await self._send_message_reply_streaming(message, stream_generator)
                else:
                    response_text = await self._generate_response(llm_messages)
                    sent_message = await self._send_message_reply(message, response_text) if response_text else None

                # Collect full response for logging and processing
                response = ""
                if sent_message:
                    response = sent_message.content

                if response:
                    # Remove username prefix in multi-user mode
                    if not is_dm:
                        response = self.strip_username_prefix(response)

                    # Check moderation on bot response
                    if self.moderation:
                        is_safe, reason = await self.moderation.check_content(response)
                        if not is_safe:
                            print(f"\n⚠️  MODERATION TRIGGERED for bot response:")
                            print(f"   Reason: {reason}")
                            print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")

                            response = "I apologize, but I cannot provide that response as it may contain inappropriate content."
                            # Edit the message with moderation override
                            if sent_message:
                                await sent_message.edit(content=response)

                    conv_manager.add_exchange(user_message, response, username)
                    
                    # Log for owner only
                    if self.data_logger.chat_logging_enabled and self.is_owner(message.author):
                        log_content = raw_content if not images else f"{raw_content or 'Image sent'}"
                        self.data_logger.log_chat(
                            user_message=log_content,
                            assistant_response=response,
                            system_prompt=self._resolve_system_prompt()
                        )
                    
                    # Cache for RLHF
                    if self.data_logger.rlhf_logging_enabled and self.is_owner(message.author) and sent_message:
                        self.message_cache[sent_message.id] = {
                            'user_message': raw_content if not images else f"{raw_content or 'Image sent'}",
                            'bot_response': response,
                            'username': message.author.name
                        }
                else:
                    await message.reply("❌ Failed to generate response. Please check if the LLM server is running.")
        
        finally:
            self.processing.discard(message.channel.id)
    
    def _setup_commands(self):
        
        @self.bot.tree.command(name="chat", description="Chat with your AI")
        @app_commands.describe(message="What do you want to say?")
        async def chat_command(interaction: discord.Interaction, message: str):
            
            if interaction.channel_id in self.processing:
                await interaction.response.send_message(
                    "⏳ Already processing a message. Please wait...",
                    ephemeral=True
                )
                return
            
            is_dm = isinstance(interaction.channel, discord.DMChannel)
            if not self.is_user_whitelisted(interaction.user, is_dm):
                await interaction.response.send_message(
                    "❌ You are not whitelisted to use this bot.",
                    ephemeral=True
                )
                return
            
            self.processing.add(interaction.channel_id)
            
            try:
                await interaction.response.defer(thinking=True)
                raw_message = message
                
                if self.moderation:
                    is_safe, reason = await self.moderation.check_content(message)
                    if not is_safe:
                        print(f"\n⚠️  MODERATION TRIGGERED for user {interaction.user.name} (slash command):")
                        print(f"   Reason: {reason}")
                        print(f"   Message: {message[:100]}{'...' if len(message) > 100 else ''}")
                        
                        embed = discord.Embed(
                            title="⚠️ Content Moderation",
                            description=f"Your message was flagged for: {reason}\n\nPlease keep the conversation appropriate.",
                            color=discord.Color.red()
                        )
                        await interaction.followup.send(embed=embed, ephemeral=True)
                        return
                
                conv_manager = self._get_conversation_manager(interaction.channel_id, is_dm)
                username = interaction.user.display_name  # Always capture username, even in DMs
                personalities = self.personality_manager.get_personalities(interaction.channel_id)
                llm_messages = conv_manager.get_messages_for_llm(message, username, personalities)

                # Use streaming or non-streaming based on config
                if self.config.get('enable_streaming', True):
                    stream_generator = self._generate_response_stream(llm_messages)
                    sent_message = await self._send_interaction_response_streaming(interaction, stream_generator)
                else:
                    response_text = await self._generate_response(llm_messages)
                    sent_message = await self._send_interaction_response(interaction, response_text) if response_text else None

                # Collect full response for logging and processing
                response = ""
                if sent_message:
                    response = sent_message.content

                if response:
                    if not is_dm:
                        response = self.strip_username_prefix(response)

                    if self.moderation:
                        is_safe, reason = await self.moderation.check_content(response)
                        if not is_safe:
                            print(f"\n⚠️  MODERATION TRIGGERED for bot response (slash command):")
                            print(f"   Reason: {reason}")
                            print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")

                            response = "I apologize, but I cannot provide that response as it may contain inappropriate content."
                            # Edit the message with moderation override
                            if sent_message:
                                await sent_message.edit(content=response)

                    conv_manager.add_exchange(message, response, username)
                    
                    if self.data_logger.chat_logging_enabled and self.is_owner(interaction.user):
                        self.data_logger.log_chat(
                            user_message=raw_message,
                            assistant_response=response,
                            system_prompt=self._resolve_system_prompt()
                        )
                else:
                    embed = discord.Embed(
                        title="❌ Error",
                        description="Failed to generate response. Please check if the LLM server is running.",
                        color=discord.Color.red()
                    )
                    await interaction.followup.send(embed=embed)
            
            except Exception as e:
                logger.error(f"Error in chat command: {e}")
                try:
                    embed = discord.Embed(
                        title="❌ Error",
                        description="An error occurred while processing your request.",
                        color=discord.Color.red()
                    )
                    await interaction.followup.send(embed=embed, ephemeral=True)
                except:
                    pass
            
            finally:
                self.processing.discard(interaction.channel_id)
        
        @self.bot.tree.command(name="clear", description="Clear conversation history")
        async def clear_command(interaction: discord.Interaction):
            
            is_dm = isinstance(interaction.channel, discord.DMChannel)
            if not self.is_user_whitelisted(interaction.user, is_dm):
                await interaction.response.send_message(
                    "❌ You are not whitelisted to use this bot.",
                    ephemeral=True
                )
                return
            
            if interaction.channel_id in self.conversations:
                self.conversations[interaction.channel_id].clear()
                
                embed = discord.Embed(
                    title="🗑️ History Cleared",
                    description="Conversation history has been reset for this channel.",
                    color=discord.Color.blue()
                )
                await interaction.response.send_message(embed=embed)
            else:
                await interaction.response.send_message(
                    "No conversation history to clear.",
                    ephemeral=True
                )
        
        @self.bot.tree.command(name="status", description="Check bot status")
        async def status_command(interaction: discord.Interaction):
            
            llm_status = "✅ Connected" if self.llm_provider.test_connection() else "❌ Disconnected"
            is_dm = isinstance(interaction.channel, discord.DMChannel)
            mode = "Single User (DM)" if is_dm else "Multi-User (Server)"
            is_whitelisted = self.is_user_whitelisted(interaction.user, is_dm)
            access_status = "✅ Whitelisted" if is_whitelisted else "❌ Not whitelisted"
            is_owner = self.is_owner(interaction.user)
            
            model_name = self.config.get('model_name', '').lower()
            vision_support = "✅ Yes" if ('vision' in model_name or 'llava' in model_name or 'bakllava' in model_name or 'minicpm' in model_name) else "❓ Unknown"
            
            config_name = self.config.get('_config_name', 'unknown')
            
            embed = discord.Embed(
                title="🤖 Bot Status",
                description=f"Configuration: **{config_name}**",
                color=discord.Color.green() if llm_status == "✅ Connected" else discord.Color.red()
            )
            
            embed.add_field(name="LLM Provider", value=self.config['llm_provider'].replace('_', ' ').title(), inline=True)
            embed.add_field(name="Model", value=self.config['model_name'], inline=True)
            embed.add_field(name="LLM Status", value=llm_status, inline=True)
            embed.add_field(name="Vision Support", value=vision_support, inline=True)
            embed.add_field(name="Moderation", value="✅ Enabled" if self.config.get('moderation_enabled') else "❌ Disabled", inline=True)
            embed.add_field(name="Mode", value=mode, inline=True)
            embed.add_field(name="Your Access", value=access_status, inline=True)
            embed.add_field(name="Context Window", value=f"{self.config.get('context_window_size', 10)} messages", inline=True)
            
            if is_owner:
                data_status = []
                if self.config.get('chat_logging_enabled'):
                    data_status.append("📝 Chat Logging")
                if self.config.get('rlhf_logging_enabled'):
                    data_status.append("👍 RLHF Logging")
                
                if data_status:
                    embed.add_field(name="Data Collection", value="\n".join(data_status), inline=True)
            
            if interaction.channel_id in self.conversations:
                conv_manager = self.conversations[interaction.channel_id]
                history_size = len(conv_manager.history)
                embed.add_field(name="Channel History", value=f"{history_size} messages", inline=True)
            
            embed.add_field(name="DM Whitelist", value=f"{len(self.dm_whitelist)} users", inline=True)
            embed.add_field(name="Server Whitelist", value=f"{len(self.server_whitelist)} users", inline=True)
            
            embed.set_footer(text=f"Uptime: {self._get_uptime()}")
            
            await interaction.response.send_message(embed=embed)
        
        @self.bot.tree.command(name="help", description="How to use this bot")
        async def help_command(interaction: discord.Interaction):
            
            is_dm = isinstance(interaction.channel, discord.DMChannel)
            is_whitelisted = self.is_user_whitelisted(interaction.user, is_dm)
            is_owner = self.is_owner(interaction.user)
            config_name = self.config.get('_config_name', 'unknown')
            
            embed = discord.Embed(
                title="📱 Chat with your PC's AI from anywhere!",
                description=f"I'm connected to the AI running on your computer.\nConfig: **{config_name}**",
                color=discord.Color.blue()
            )
            
            if is_whitelisted:
                embed.add_field(
                    name="✅ Your Access",
                    value=f"You are whitelisted for {'DM' if is_dm else 'server'} use!",
                    inline=False
                )
            else:
                embed.add_field(
                    name="❌ Access Denied",
                    value="You are not whitelisted to use this bot.\nPlease contact the bot owner.",
                    inline=False
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return
            
            if is_dm:
                embed.add_field(
                    name="💬 In DMs",
                    value=(
                        "• Just type! No commands needed\n"
                        "• Send images with vision models\n"
                        "• I'll remember our conversation\n"
                        "• Use `/clear` to start fresh\n"
                        "• Single-user mode (just you and me)"
                    ),
                    inline=False
                )
            else:
                embed.add_field(
                    name="💬 In Servers",
                    value=(
                        "• Mention me: @bot your message\n"
                        "• Reply to my messages\n"
                        "• Send images with vision models\n"
                        "• Or use `/chat` command\n"
                        "• Multi-user mode (I see who's talking)"
                    ),
                    inline=False
                )
            
            model_name = self.config.get('model_name', '').lower()
            if 'vision' in model_name or 'llava' in model_name or 'bakllava' in model_name or 'minicpm' in model_name:
                embed.add_field(
                    name="🖼️ Vision Support",
                    value=(
                        "• Your model supports images!\n"
                        "• Just attach an image to your message\n"
                        "• Ask questions about the image\n"
                        "• Works in both DMs and servers"
                    ),
                    inline=False
                )
            
            if is_owner and (self.config.get('chat_logging_enabled') or self.config.get('rlhf_logging_enabled')):
                data_features = []
                if self.config.get('chat_logging_enabled'):
                    data_features.append(f"• 📝 Chat logging: ON (saving to chat_logs/{config_name}_*.jsonl)")
                if self.config.get('rlhf_logging_enabled'):
                    data_features.append(f"• 👍 RLHF: React with 👍/👎 to log feedback (rlhf_logs/{config_name}_*.jsonl)")
                
                embed.add_field(
                    name="📊 Data Collection (Owner Only)",
                    value="\n".join(data_features),
                    inline=False
                )
            
            embed.add_field(
                name="📝 How to DM the Bot",
                value=(
                    "• First add me to a server\n"
                    "• Right-click me in the member list\n"
                    "• Select 'Message' to start a DM"
                ),
                inline=False
            )
            
            embed.add_field(
                name="⚙️ Commands",
                value=(
                    "• `/chat <message>` - Chat with command\n"
                    "• `/clear` - Reset conversation\n"
                    "• `/status` - Check connection\n"
                    "• `/help` - This message"
                ),
                inline=False
            )
            
            embed.add_field(
                name="📝 Current Settings",
                value=(
                    f"• Model: {self.config['model_name']}\n"
                    f"• Memory: {self.config.get('context_window_size', 10)} messages\n"
                    f"• Mode: {'Single-user (DM)' if is_dm else 'Multi-user (Server)'}"
                ),
                inline=False
            )
            
            await interaction.response.send_message(embed=embed)

        # Prefix Commands for Personality Management
        @self.bot.command(name="personality")
        async def personality_command(ctx: commands.Context, *args):
            """
            Manage personality instructions for the channel.
            Usage:
              !personality <slot> <text> - Set a personality
              !personality clear <slot> - Clear a slot
              !personality clearall - Clear all personalities
              !personality list - List all personalities
            """
            is_dm = isinstance(ctx.channel, discord.DMChannel)
            if not self.is_user_whitelisted(ctx.author, is_dm):
                await ctx.send("❌ You are not whitelisted to use this bot.")
                return

            channel_id = ctx.channel.id

            # Handle subcommands
            if not args:
                await ctx.send("❌ Usage: `!personality <slot> <text>` | `!personality clear <slot>` | `!personality clearall` | `!personality list`")
                return

            first_arg = args[0].lower()

            # List personalities
            if first_arg == "list":
                personalities_str = self.personality_manager.list_personalities(channel_id)
                embed = discord.Embed(
                    title="📋 Current Personalities",
                    description=personalities_str,
                    color=discord.Color.blue()
                )
                await ctx.send(embed=embed)
                return

            # Clear all personalities
            if first_arg == "clearall":
                success, message = self.personality_manager.clear_all_personalities(channel_id)
                await ctx.send(message)
                return

            # Clear a specific personality
            if first_arg == "clear":
                if len(args) < 2:
                    await ctx.send("❌ Usage: `!personality clear <slot>`")
                    return
                try:
                    slot_num = int(args[1])
                    success, message = self.personality_manager.clear_personality(channel_id, slot_num)
                    await ctx.send(message)
                except ValueError:
                    await ctx.send(f"❌ Slot must be a number (0-{PersonalityManager.MAX_SLOTS - 1})")
                return

            # Set a personality (slot and text)
            if len(args) < 2:
                await ctx.send("❌ Usage: `!personality <slot> <text>`")
                return

            try:
                slot_num = int(first_arg)
                text = " ".join(args[1:])
                success, message = self.personality_manager.set_personality(channel_id, slot_num, text)
                await ctx.send(message)
            except ValueError:
                await ctx.send(f"❌ Slot must be a number (0-{PersonalityManager.MAX_SLOTS - 1})")
            return

    async def _generate_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Run LLM generation in executor to avoid blocking"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.llm_provider.generate_response,
                messages
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    async def _send_message_reply(self, message: discord.Message, response: str) -> Optional[discord.Message]:
        """Send response with Discord's 2000 character limit handling"""
        MAX_LENGTH = 1900
        
        try:
            if len(response) <= MAX_LENGTH:
                return await message.reply(response)
            else:
                chunks = [response[i:i+MAX_LENGTH] for i in range(0, len(response), MAX_LENGTH)]
                
                first_msg = None
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        first_msg = await message.reply(chunk)
                    else:
                        await message.channel.send(chunk)
                return first_msg
        except Exception as e:
            logger.error(f"Error sending message reply: {e}")
            return None
    
    async def _send_interaction_response(self, interaction: discord.Interaction, response: str) -> Optional[discord.Message]:
        """Send slash command response with character limit handling"""
        MAX_LENGTH = 1900

        try:
            if len(response) <= MAX_LENGTH:
                return await interaction.followup.send(response)
            else:
                chunks = [response[i:i+MAX_LENGTH] for i in range(0, len(response), MAX_LENGTH)]

                first_msg = None
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        first_msg = await interaction.followup.send(chunk)
                    else:
                        await interaction.channel.send(chunk)
                return first_msg
        except Exception as e:
            logger.error(f"Error sending interaction response: {e}")
            return None

    async def _generate_response_stream(self, messages: List[Dict[str, Any]]):
        """Run LLM generation in executor and yield chunks as they arrive"""
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def collect_chunks_in_executor():
            """Run in executor thread to collect chunks from generator"""
            try:
                generator = self.llm_provider.generate_response_stream(messages)
                for chunk in generator:
                    # Use thread-safe method to put items in the queue
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                # Signal end of stream
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # Start collecting chunks in executor
        loop.run_in_executor(None, collect_chunks_in_executor)

        # Yield chunks from queue as they arrive
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if chunk:
                yield chunk

    async def _send_message_reply_streaming(self, message: discord.Message, stream_generator) -> Optional[discord.Message]:
        """Send response with streaming, editing message as chunks arrive"""
        MAX_LENGTH = 1900

        try:
            full_response = ""
            sent_message = None
            last_edit = 0
            min_edit_interval = 0.5  # Minimum seconds between edits to avoid rate limiting

            async for chunk in stream_generator:
                full_response += chunk

                # Edit message every few chunks or when approaching limit
                current_time = asyncio.get_event_loop().time()
                should_edit = (current_time - last_edit) >= min_edit_interval or len(full_response) % 500 == 0

                if should_edit:
                    # Check if we need to split into multiple messages
                    if len(full_response) <= MAX_LENGTH:
                        if sent_message is None:
                            sent_message = await message.reply(full_response)
                        else:
                            await sent_message.edit(content=full_response)
                        last_edit = current_time
                    else:
                        # Already too long, start new message
                        break

            # Final edit with complete response
            if sent_message and len(full_response) <= MAX_LENGTH:
                await sent_message.edit(content=full_response)
            elif len(full_response) > MAX_LENGTH:
                # Split into chunks
                if sent_message is None:
                    sent_message = await message.reply(full_response[:MAX_LENGTH])
                else:
                    await sent_message.edit(content=full_response[:MAX_LENGTH])

                # Send remaining chunks
                remaining = full_response[MAX_LENGTH:]
                while remaining:
                    chunk = remaining[:MAX_LENGTH]
                    await message.channel.send(chunk)
                    remaining = remaining[MAX_LENGTH:]

            return sent_message
        except Exception as e:
            logger.error(f"Error sending streaming message reply: {e}")
            return None

    async def _send_interaction_response_streaming(self, interaction: discord.Interaction, stream_generator) -> Optional[discord.Message]:
        """Send slash command response with streaming, editing message as chunks arrive"""
        MAX_LENGTH = 1900

        try:
            full_response = ""
            sent_message = None
            last_edit = 0
            min_edit_interval = 0.5  # Minimum seconds between edits to avoid rate limiting

            async for chunk in stream_generator:
                full_response += chunk

                # Edit message every few chunks or when approaching limit
                current_time = asyncio.get_event_loop().time()
                should_edit = (current_time - last_edit) >= min_edit_interval or len(full_response) % 500 == 0

                if should_edit:
                    # Check if we need to split into multiple messages
                    if len(full_response) <= MAX_LENGTH:
                        if sent_message is None:
                            sent_message = await interaction.followup.send(full_response)
                        else:
                            await sent_message.edit(content=full_response)
                        last_edit = current_time
                    else:
                        # Already too long, start new message
                        break

            # Final edit with complete response
            if sent_message and len(full_response) <= MAX_LENGTH:
                await sent_message.edit(content=full_response)
            elif len(full_response) > MAX_LENGTH:
                # Split into chunks
                if sent_message is None:
                    sent_message = await interaction.followup.send(full_response[:MAX_LENGTH])
                else:
                    await sent_message.edit(content=full_response[:MAX_LENGTH])

                # Send remaining chunks
                remaining = full_response[MAX_LENGTH:]
                while remaining:
                    chunk = remaining[:MAX_LENGTH]
                    await interaction.channel.send(chunk)
                    remaining = remaining[MAX_LENGTH:]

            return sent_message
        except Exception as e:
            logger.error(f"Error sending streaming interaction response: {e}")
            return None

    def _get_uptime(self) -> str:
        if not hasattr(self, 'start_time'):
            return "Just started"
        
        delta = datetime.now() - self.start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    async def start(self):
        """Start the Discord bot"""
        self.start_time = datetime.now()
        
        try:
            await self.bot.start(self.config['discord_token'])
        except discord.LoginFailure:
            print("\n❌ Invalid Discord token! Please check your configuration.")
            raise
        except Exception as e:
            print(f"\n❌ Failed to start bot: {e}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        print("\n🛑 Shutting down bot...")
        await self.bot.close()
