"""
Configuration Manager - Setup wizard with all options
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any, List
import getpass
from llm_providers import OllamaProvider, LMStudioProvider
from utils import Colors, clear_screen


class ConfigManager:
    """Configuration wizard - handles all setup options"""
    
    CONFIG_DIR = Path("configs")
    CONFIG_SUFFIX = "_config.json"
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_LMSTUDIO_URL = "http://localhost:1234"
    
    def __init__(self):
        self.CONFIG_DIR.mkdir(exist_ok=True)
    
    def list_configs(self) -> List[str]:
        """List all available config files"""
        configs = []
        for file in self.CONFIG_DIR.glob(f"*{self.CONFIG_SUFFIX}"):
            name = file.stem.replace("_config", "")
            configs.append(name)
        return sorted(configs)
    
    def config_exists(self, name: str = None) -> bool:
        """Check if a specific config exists, or if any configs exist"""
        if name:
            config_path = self.CONFIG_DIR / f"{name}{self.CONFIG_SUFFIX}"
            return config_path.exists()
        else:
            return len(self.list_configs()) > 0
    
    def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a specific named config"""
        config_path = self.CONFIG_DIR / f"{name}{self.CONFIG_SUFFIX}"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Remove deprecated settings
                config.pop('multi_user_mode', None)
                config.pop('ollama_context_tokens', None)
                config['_config_name'] = name
                return config
        except Exception as e:
            print(Colors.red(f"❌ Couldn't load config '{name}': {e}"))
            return None
    
    def save_config(self, config: Dict[str, Any], name: str) -> bool:
        """Save settings with a specific name"""
        config_path = self.CONFIG_DIR / f"{name}{self.CONFIG_SUFFIX}"
        try:
            config_copy = config.copy()
            config_copy.pop('_config_name', None)
            
            with open(config_path, 'w') as f:
                json.dump(config_copy, f, indent=2)
            return True
        except Exception as e:
            print(Colors.red(f"❌ Couldn't save config '{name}': {e}"))
            return False
    
    def delete_config(self, name: str) -> bool:
        """Delete a specific config file"""
        config_path = self.CONFIG_DIR / f"{name}{self.CONFIG_SUFFIX}"
        try:
            if config_path.exists():
                config_path.unlink()
                return True
            return False
        except Exception as e:
            print(Colors.red(f"❌ Couldn't delete config '{name}': {e}"))
            return False
    
    def choose_config(self) -> Optional[str]:
        """Let user choose from existing configs"""
        configs = self.list_configs()
        
        if not configs:
            return None
        
        print(Colors.yellow("Available configurations:\n"))
        for i, name in enumerate(configs, 1):
            cfg = self.load_config(name)
            if cfg:
                model = cfg.get('model_name', 'unknown')
                provider = cfg.get('llm_provider', 'unknown').replace('_', ' ').title()
                print(f"  [{i}] {Colors.cyan(name)} - {provider}, {model}")
            else:
                print(f"  [{i}] {name}")
        
        print()
        while True:
            choice = input(f"Select config (1-{len(configs)}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(configs):
                    return configs[idx]
            except ValueError:
                pass
            print(Colors.red(f"Please enter a number between 1 and {len(configs)}"))
    
    def get_config_name(self, suggested: str = None) -> str:
        """Get a valid config name from user"""
        print(Colors.yellow("\nName your configuration:\n"))
        print(Colors.cyan("This helps you manage multiple bots with different settings."))
        print(Colors.cyan("Examples: 'gaming-bot', 'work-assistant', 'creative-writer'\n"))
        
        while True:
            if suggested:
                name = input(f"Config name (default: {suggested}): ").strip()
                if not name:
                    name = suggested
            else:
                name = input("Config name: ").strip()
            
            if not name:
                print(Colors.red("Name cannot be empty"))
                continue
            
            # Remove invalid characters
            cleaned = "".join(c for c in name if c.isalnum() or c in "-_")
            if cleaned != name:
                print(Colors.yellow(f"Invalid characters removed: '{cleaned}'"))
                name = cleaned
            
            if not name:
                print(Colors.red("Name must contain at least one valid character"))
                continue
            
            if self.config_exists(name):
                print(Colors.yellow(f"\nConfig '{name}' already exists."))
                overwrite = input("Overwrite? (y/n): ").strip().lower()
                if overwrite == 'y':
                    return name
                continue
            
            return name
    
    def get_whitelisted_users(self, context: str) -> List[str]:
        """Get list of whitelisted users for DMs or servers"""
        users = []
        
        print(Colors.cyan(f"\nAdd users who can use the bot {context}:"))
        print(Colors.cyan("Enter Discord usernames (without @), one at a time."))
        print(Colors.cyan("Press Enter with empty input when done.\n"))
        
        while True:
            username = input(f"Username (or Enter to finish): ").strip().lower()
            if not username:
                break
            
            username = username.replace('@', '').replace(' ', '')
            
            if username and username not in users:
                users.append(username)
                print(Colors.green(f"  ✅ Added: {username}"))
            elif username in users:
                print(Colors.yellow(f"  ⚠️ Already added: {username}"))
        
        if users:
            print(Colors.green(f"\n✅ Added {len(users)} user(s) to whitelist"))
        
        return users
    
    def setup_wizard(self, config_name: str = None) -> Optional[Dict[str, Any]]:
        """Complete setup wizard with all options"""
        
        config = {}
        
        print(Colors.blue("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
        print(Colors.green("      🧙 Setup Wizard"))
        print(Colors.blue("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"))
        
        if not config_name:
            config_name = self.get_config_name("my-bot")
        
        print(Colors.green(f"\n✅ Creating config: '{config_name}'\n"))
        
        # Step 1: Choose LLM Provider
        print(Colors.yellow("Step 1: Which AI software are you using?\n"))
        print("  [1] 🦙 Ollama")
        print("  [2] 🖥️  LM Studio\n")
        
        while True:
            choice = input("Select (1-2): ").strip()
            if choice == "1":
                config['llm_provider'] = "ollama"
                config['llm_base_url'] = self.DEFAULT_OLLAMA_URL
                print(Colors.green("\n✅ Using Ollama\n"))
                break
            elif choice == "2":
                config['llm_provider'] = "lm_studio"
                config['llm_base_url'] = self.DEFAULT_LMSTUDIO_URL
                print(Colors.green("\n✅ Using LM Studio\n"))
                break
            else:
                print(Colors.red("Please enter 1 or 2"))
        
        # Ask about custom URL
        if config['llm_provider'] == "ollama":
            print(Colors.cyan("📍 Finding Ollama's server URL:"))
            print(Colors.cyan("   - Run 'ollama serve' in a terminal"))
            print(Colors.cyan("   - Look for 'Ollama is running on ...' message"))
            print(Colors.cyan("   - Copy the full URL shown (e.g., http://localhost:11434)\n"))
        else:
            print(Colors.cyan("📍 Finding LM Studio's server URL:"))
            print(Colors.cyan("   - Open LM Studio → Developer tab"))
            print(Colors.cyan("   - Look at the server URL shown"))
            print(Colors.cyan("   - Copy the exact URL (e.g., http://localhost:1234)\n"))
        
        custom_url = input("Enter the URL you see (or press Enter for default): ").strip()
        if custom_url:
            config['llm_base_url'] = custom_url if custom_url.startswith('http') else f"http://{custom_url}"
        else:
            print(Colors.yellow(f"Using default URL: {config['llm_base_url']}"))
        
        # Step 2: Test Connection and Get Models
        print(Colors.yellow("\nStep 2: Connecting to your AI...\n"))
        
        provider_name = "Ollama" if config['llm_provider'] == "ollama" else "LM Studio"
        
        if config['llm_provider'] == "ollama":
            provider = OllamaProvider(config['llm_base_url'])
        else:
            provider = LMStudioProvider(config['llm_base_url'])
        
        print(f"🔍 Connecting to {provider_name}...")
        
        if not provider.test_connection():
            print(Colors.red(f"\n❌ Can't connect to {provider_name} at {config['llm_base_url']}"))
            
            if config['llm_provider'] == "ollama":
                print("\nMake sure Ollama is running:")
                print(Colors.cyan("  1. Open a new terminal"))
                print(Colors.cyan("  2. Run: ollama serve"))
                print(Colors.cyan("  3. Try setup again\n"))
            else:
                print("\nMake sure LM Studio server is running:")
                print(Colors.cyan("  1. Open LM Studio"))
                print(Colors.cyan("  2. Go to Developer tab"))
                print(Colors.cyan("  3. Start the server"))
                print(Colors.cyan("  4. Try setup again\n"))
            return None
        
        print(Colors.green(f"✅ Connected to {provider_name}!\n"))
        
        print(f"📚 Getting available models...")
        models = provider.list_models()
        
        if not models:
            print(Colors.red("\n❌ No models found!"))
            if config['llm_provider'] == "ollama":
                print("\nRun: ollama pull llama3.2")
            else:
                print("\nDownload a model in LM Studio first")
            return None
        
        # Step 3: Select Model
        print(Colors.green(f"\n✅ Found {len(models)} model(s)!\n"))
        print(Colors.yellow("Step 3: Select your model:\n"))
        
        for i, model in enumerate(models[:20], 1):
            print(f"  [{i}] {model}")
        print()
        
        while True:
            try:
                choice = input(f"Select (1-{min(len(models), 20)}): ").strip()
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(models):
                    config['model_name'] = models[model_idx]
                    print(Colors.green(f"\n✅ Selected: {config['model_name']}\n"))
                    break
            except ValueError:
                print(Colors.red("Please enter a valid number"))
        
        # Step 4: Discord Bot Token
        print(Colors.yellow("Step 4: Discord Bot Token\n"))
        print("Get your bot token from Discord:")
        print(Colors.cyan("  1. Go to: https://discord.com/developers/applications"))
        print(Colors.cyan("  2. Create New Application or select existing"))
        print(Colors.cyan("  3. Go to Bot section"))
        print(Colors.cyan("  4. Reset Token and copy it\n"))
        
        while True:
            print("Paste your Discord bot token:")
            print(Colors.cyan("(Your input will be hidden for security)"))
            token = getpass.getpass("Token: ").strip()
            
            if token and len(token) > 50:
                config['discord_token'] = token
                print(Colors.green("✅ Discord token received and saved!\n"))
                break
            else:
                print(Colors.red("Invalid token. Discord tokens are usually 70+ characters. Try again.\n"))
        
        # Step 5: User Whitelisting
        print(Colors.yellow("Step 5: User Whitelisting (Security)\n"))
        print(Colors.cyan("🔒 This controls who can use your bot and GPU resources.\n"))
        
        print("Enter YOUR Discord username (you'll be automatically whitelisted):")
        print(Colors.cyan("(This is your Discord username, not your display name)"))
        
        while True:
            owner_username = input("Your username: ").strip().lower()
            if owner_username:
                owner_username = owner_username.replace('@', '').replace(' ', '')
                config['owner_username'] = owner_username
                print(Colors.green(f"\n✅ You ({owner_username}) are whitelisted for both DMs and server use!\n"))
                break
            else:
                print(Colors.red("Please enter your username"))
        
        config['dm_whitelist'] = [owner_username]
        config['server_whitelist'] = [owner_username]
        
        print("Would you like to whitelist other users?")
        print(Colors.cyan("(Only trusted users should be whitelisted - they'll use your GPU)\n"))
        
        whitelist_others = input("Add other users? (y/n): ").strip().lower()
        
        if whitelist_others == 'y':
            print(Colors.yellow("\n📱 DM Whitelist"))
            print("Users who can privately message the bot:")
            dm_users = self.get_whitelisted_users("in DMs")
            config['dm_whitelist'].extend(dm_users)
            
            print(Colors.yellow("\n🏢 Server Whitelist"))
            print("Users who can use the bot in servers:")
            server_users = self.get_whitelisted_users("in servers")
            config['server_whitelist'].extend(server_users)
            
            print(Colors.cyan("\n📋 Whitelist Summary:"))
            print(f"  DM Access: {', '.join(config['dm_whitelist'])}")
            print(f"  Server Access: {', '.join(config['server_whitelist'])}")
        else:
            print(Colors.green("\n✅ Only you can use the bot (most secure)\n"))
        
        # Step 6: Moderation
        print(Colors.yellow("Step 6: Content Moderation (Optional)\n"))
        print("Enable OpenAI's free content filter?")
        print("Recommended if others will use your bot.\n")
        
        mod_choice = input("Enable moderation? (y/n): ").strip().lower()
        config['moderation_enabled'] = mod_choice == 'y'
        
        if config['moderation_enabled']:
            print(Colors.cyan("\nGet a free OpenAI API key:"))
            print(Colors.cyan("  1. Go to: https://platform.openai.com/api-keys"))
            print(Colors.cyan("  2. Create an API key\n"))
            
            print("Paste your OpenAI API key:")
            print(Colors.cyan("(Your input will be hidden for security)"))
            api_key = getpass.getpass("API Key: ").strip()
            
            if api_key:
                config['openai_api_key'] = api_key
                print(Colors.green("✅ OpenAI API key received and saved!"))
                print(Colors.green("✅ Moderation enabled!\n"))
            else:
                config['moderation_enabled'] = False
                config['openai_api_key'] = ""
                print(Colors.yellow("\n⚠️  No API key provided - moderation disabled\n"))
        else:
            config['openai_api_key'] = ""
            print(Colors.green("\n✅ Moderation disabled\n"))
        
        # Step 7: Chat Logging
        print(Colors.yellow("Step 7: Chat Logging (Optional)\n"))
        print("Save your conversations for fine-tuning datasets?")
        print(Colors.cyan("• Saves in ChatML format (industry standard)"))
        print(Colors.cyan("• Only YOUR conversations are logged (privacy)"))
        print(Colors.cyan("• Creates timestamped files in chatlogs/ folder\n"))
        
        chat_log_choice = input("Enable chat logging? (y/n): ").strip().lower()
        config['chat_logging_enabled'] = chat_log_choice == 'y'
        
        if config['chat_logging_enabled']:
            print(Colors.green("\n✅ Chat logging enabled!"))
            print(Colors.cyan("Files will be saved in: chatlogs/\n"))
        else:
            print(Colors.green("\n✅ Chat logging disabled\n"))
        
        # Step 8: RLHF Logging
        print(Colors.yellow("Step 8: RLHF Data Collection (Optional)\n"))
        print("Log thumbs up/down reactions for reinforcement learning?")
        print(Colors.cyan("• React with 👍 or 👎 to bot messages"))
        print(Colors.cyan("• Only YOUR reactions are logged (privacy)"))
        print(Colors.cyan("• Perfect for preference datasets"))
        print(Colors.cyan("• Saves in rlhf_logs/ folder\n"))
        
        rlhf_choice = input("Enable RLHF logging? (y/n): ").strip().lower()
        config['rlhf_logging_enabled'] = rlhf_choice == 'y'
        
        if config['rlhf_logging_enabled']:
            print(Colors.green("\n✅ RLHF logging enabled!"))
            print(Colors.cyan("React to messages with 👍 or 👎 to log feedback"))
            print(Colors.cyan("Files will be saved in: rlhf_logs/\n"))
        else:
            print(Colors.green("\n✅ RLHF logging disabled\n"))
        
        # Step 9: Context Window
        print(Colors.yellow("Step 9: Memory Settings\n"))
        print("How many messages should the bot remember?")
        print("(More = better context, but slower)")
        print("The bot will automatically manage the context window.\n")
        
        while True:
            try:
                context = input("Number of messages (default 10): ").strip()
                if not context:
                    config['context_window_size'] = 10
                    break
                else:
                    size = int(context)
                    if 1 <= size <= 100:
                        config['context_window_size'] = size
                        break
                    else:
                        print(Colors.red("Please enter 1-100"))
            except ValueError:
                print(Colors.red("Please enter a number"))
        
        print(Colors.green(f"\n✅ Will remember {config['context_window_size']} messages\n"))
        
        # Step 10: System Prompt
        print(Colors.yellow("Step 10: Bot Personality (Optional)\n"))
        print("Give your bot instructions or personality.")
        print("Leave empty for default assistant.\n")
        
        system_prompt = input("System prompt (or press Enter): ").strip()
        if not system_prompt:
            system_prompt = "You are a helpful AI assistant. Be friendly and concise."
        config['system_prompt'] = system_prompt
        
        print(Colors.cyan("\n📝 Note: Bot behavior differs by location:"))
        print(Colors.cyan("   • In DMs: Single-user mode, just chat naturally"))
        print(Colors.cyan("   • In Servers: Multi-user mode, mention or reply to bot"))
        print(Colors.cyan("   • Only whitelisted users can interact with the bot"))
        
        if config.get('chat_logging_enabled') or config.get('rlhf_logging_enabled'):
            print(Colors.cyan("\n📊 Data Collection Active:"))
            if config.get('chat_logging_enabled'):
                print(Colors.cyan("   • Chat logging: ON (only your messages)"))
            if config.get('rlhf_logging_enabled'):
                print(Colors.cyan("   • RLHF logging: ON (only your reactions)"))
        
        # Save configuration
        print(Colors.blue("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
        print(Colors.green("        ✅ Setup Complete!"))
        print(Colors.blue("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"))
        
        if self.save_config(config, config_name):
            print(f"Settings saved to: {Colors.cyan(f'configs/{config_name}_config.json')}")
            print(Colors.yellow("\n📝 Config Management:"))
            print(f"  • Edit manually: Open {Colors.cyan(f'configs/{config_name}_config.json')} in any text editor")
            print(f"  • Delete config: Delete the file {Colors.cyan(f'configs/{config_name}_config.json')}")
            print(f"  • Multiple bots: Create multiple configs with different names")
            print(f"  • Edit whitelist: Modify 'dm_whitelist' and 'server_whitelist' in the config file")
            
            print("\n" + Colors.cyan("Next steps:"))
            print(Colors.cyan("  1. Bot will start now"))
            print(Colors.cyan("  2. Open Discord"))
            print(Colors.cyan("  3. DM the bot or mention it in a server"))
            print(Colors.cyan("  4. Chat with your AI!"))
            
            if config.get('rlhf_logging_enabled'):
                print(Colors.cyan("\n  💡 Try reacting with 👍 or 👎 to bot messages!"))
            
            input("\nPress Enter to start...")
            
            config['_config_name'] = config_name
            
            return config
        else:
            print(Colors.yellow("\n⚠️  Couldn't save settings but bot will run"))
            input("Press Enter to continue...")
            return config
