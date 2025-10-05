# 📖 Setup Guide

This guide will walk you through setting up your Local LLM Discord Bot step by step.

## 📋 Prerequisites

Before starting, make sure you have:

- **Windows PC with Python 3.10+** - Check with `python --version`
- **Ollama or LM Studio installed** - With at least one model downloaded
- **Discord account** - To create and use the bot

## 🚀 Initial Setup

### Step 1: Start Your LLM Server First

**IMPORTANT: Do this BEFORE running main.py!**

**For Ollama Users:**
```bash
# Start Ollama server first
ollama serve

# Get a model if you don't have one
ollama pull <model-name>
```

**For LM Studio Users:**
1. Open LM Studio
2. Download a model from Discover tab if you don't have one
3. Go to Developer tab → Start Server
4. Note the server URL shown (you'll need this during setup)

### Step 2: Install the Bot

```bash
git clone https://github.com/ella0333/Local-LLM-Discord-Bot.git
cd Local-LLM-Discord-Bot
pip install -r requirements.txt
```

### Step 3: Create Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application** → Name it → **Create** → Feel free to add an icon for the bot
3. Go to **Bot** section (left sidebar)
4. Click **Reset Token** → Copy the token (you'll need this soon)
5. **IMPORTANT - Enable these intents:**
   - ✅ **MESSAGE CONTENT INTENT** 
   - ✅ **SERVER MEMBERS INTENT** 

### Step 4: Add Bot to Discord

1. Still in Developer Portal → **OAuth2** → **URL Generator**
2. Select scopes: 
   - ✅ bot
   - ✅ applications.commands
3. Select bot permissions:
   - ✅ Send Messages
   - ✅ View Channels
   - ✅ Read Message History
   - ✅ Use Slash Commands
   - ✅ Add Reactions
4. Copy the generated URL
5. Open URL → Add to your server

**To DM the bot:** After adding to a server, right-click the bot in the member list and select "Message" to start a DM conversation

### Step 5: Run Setup Wizard

```bash
python main.py
```

The setup wizard will ask you for:

1. **LLM Provider** - Choose Ollama or LM Studio
2. **Model Selection** - Pick from your installed models  
3. **Discord Bot Token** - Paste the token from Step 3
4. **Your Discord Username** - You'll be automatically whitelisted
5. **Additional Users** (optional) - Whitelist trusted users for DMs and/or servers
6. **Content Moderation** (optional) - Enable OpenAI's free content filter
7. **Chat Logging** (optional) - Save conversations for training datasets (owner only)
8. **RLHF Logging** (optional) - Log feedback reactions (owner only)
9. **Memory Size** - How many messages to remember (default 10)
10. **Bot Personality** - Custom system prompt (optional)

Your settings are automatically saved!

## 💬 Usage Guide

### Starting the Bot

After setup, the bot starts automatically. To start it again later:

```bash
python main.py
# Choose [1] to use saved settings
# Or [2] to run setup again
```

### Natural Chat (No Commands Needed!)

**In DMs:** 
- First add the bot to a server (see Step 4)
- Right-click the bot in the server member list
- Select "Message" to open DM
- Just type your message!
- **Note: Only whitelisted users can DM the bot**

```
You: Hey, how are you?
Bot: I'm doing great! How can I help you today?
```

**In Servers:** Multiple ways to chat
- **Note: Only whitelisted users can use the bot in servers**

```
# Mention the bot
You: @BotName what's the weather like?

# Reply to any bot message
You: [replying to bot] tell me more about that

# Or use the slash command
You: /chat what's the weather like?
```

**With Vision Models:**
- If using a vision-capable model, you can attach images
- The bot will analyze the images and respond
- Works in both DMs and servers

**With Reasoning Models**
- The bot works seamlessly with reasoning models
- Models still think through problems internally (chain-of-thought reasoning happens)
- The thinking process is automatically hidden from Discord responses
- **You can see the full reasoning in your server logs:**
  - **Ollama:** Check the terminal where `ollama serve` is running
  - **LM Studio:** Check the Developer tab console/logs
- Discord users only see the clean, final answer
- This keeps responses concise while maintaining the model's reasoning capabilities

**If thinking appears in Discord messages:**
- The model might use a non-standard tag format
- See the [Troubleshooting Guide](TROUBLESHOOTING.md#-reasoning-model-issues) for solutions
- Email Gabriella@Kryptive.com with the model name and output to add support

### Discord Commands

- `/chat [message]` - Chat with your AI (alternative to natural chat)
- `/clear` - Clear conversation history for this channel
- `/status` - Check connection status, model info, and current mode
- `/help` - Show help and current settings

### Stopping the Bot

Press `Ctrl+C` in the terminal window running the bot.

## ⚙️ Configuration Details

### User Whitelisting

The bot includes a whitelist system to control access:
- **DM Whitelist**: Controls who can privately message the bot
- **Server Whitelist**: Controls who can use the bot in servers
- During setup, you're automatically added to both lists
- You can optionally add trusted users during setup
- To modify the whitelist later, edit your config file in `configs/`
- This is intentionally config-only for security - no Discord commands to add users

### Context Window Management

The bot uses a smart shifting context window:
- Keeps your configured number of recent messages (e.g., last 10 messages)
- System prompt is preserved
- Older messages automatically drop off as new ones come in
- Each channel has its own separate context
- **Note:** If the model's context limit is reached before your message limit, the LLM provider will handle it automatically (Ollama/LM Studio will truncate older messages to stay within the model's capabilities)

### Single vs Multi-User Modes

**Automatic Detection:**
- **DMs** → Single-user mode (personal conversation)
- **Servers** → Multi-user mode (includes usernames in context for LLM)

**How Multi-User Mode Works:**
- In server channels, the bot sees who's talking to it
- Each user's message is prefixed with `[username]:` before being sent to the LLM
- This helps the AI understand who is speaking in group conversations
- The bot's system prompt is automatically updated to explain this format
- The bot is instructed NOT to use a prefix for its own responses
- This creates natural conversations where the bot responds directly without mimicking the username prefix format

**Example of what the bot sees in multi-user mode:**
```
[Alice]: Hey bot, what's the capital of France?
[Bot responds]: The capital of France is Paris.
[Bob]: Can you tell us more about it?
[Bot responds]: Paris is known as the "City of Light"...
```

No configuration needed! The bot automatically adapts based on whether it's in a DM or server channel.

### Memory Settings

- **Default:** 10 messages
- **Range:** 1-100 messages
- **Note:** More messages = better context but slower responses
- The bot respects model context limits automatically if the message limit set exeeds it

### Model Temperature and Parameters

**You can adjust your model's behavior through the provider settings:**

**For Ollama Users:**
- Create or modify a Modelfile to set temperature and other parameters
- Example:
  ```bash
  # Create a custom model with different temperature
  echo "FROM llama3.2
  PARAMETER temperature 0.8
  PARAMETER top_p 0.9
  PARAMETER top_k 40" > Modelfile
  
  ollama create my-custom-model -f Modelfile
  ```
- Temperature range: 0.0 (deterministic) to 2.0 (creative)
- See [Ollama Modelfile docs](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) for all parameters

**For LM Studio Users:**
- Adjust parameters directly in the LM Studio interface
- Go to the **Developer** tab → **Server Model Settings**
- Available settings:
  - **Temperature**: Controls randomness (0.0 = focused, 1.0 = creative)
  - **Top P**: Nucleus sampling (0.1 to 1.0)
  - **Top K**: Limits vocabulary (1 to 100+)
  - **Repeat Penalty**: Reduces repetition (1.0 to 1.5)
  - **Max Tokens**: Response length limit
- Changes apply immediately without restarting the bot

### Content Moderation

When moderation is enabled using OpenAI's API:
- All messages are scanned for inappropriate content before being sent to your LLM
- Bot responses are also scanned before being sent to users
- **Important:** If OpenAI's API service goes down, moderation will be disabled temporarily, but messages will still be able to be sent (the bot fails open to maintain functionality)
- Moderation uses OpenAI's free content moderation endpoint

### Reloading Configuration Changes

**After Editing Configuration:**
If you manually edit your config file (in `configs/` folder), you need to restart the bot to reload the changes:

1. **Stop the bot**:
   - Press `Ctrl+C` in the terminal running the bot

2. **Restart the bot**:
   - Run `python main.py` again
   - Select your config to reload

The bot will load your updated configuration on startup.

## 📊 Data Collection Features

### Chat Logging

When enabled, the bot saves conversations in ChatML format for fine-tuning:
- **Only available for the main user** (owner) for privacy reasons
- Saved in `chatlogs/` folder as JSONL files
- Each session creates a new timestamped file with config name
- File naming: `chatlog_[config_name]_[timestamp].jsonl`
- Format: Standard ChatML for easy use with training tools like unsloth
- Excludes default system prompts (only logs custom ones)
- **Files are never overwritten** - each bot run creates a new file

### RLHF Data Collection

Collect human feedback for reinforcement learning:
- **Only the main user's reactions are logged** for privacy reasons
- React with 👍 or 👎 to any bot message to log feedback
- **Note:** RLHF doesn't work with slash commands as there is no message to react to
- Saved in `rlhf_logs/` folder as JSONL files
- File naming: `rlhf_[config_name]_[timestamp].jsonl`
- Only messages with reactions are logged
- Includes whether response was rated good or bad
- Perfect for creating preference datasets
- **Files are never overwritten** - each bot run creates a new file

**Important:** Other whitelisted users' conversations and reactions are never logged, ensuring their privacy if they are dming with bot.

## 🔒 Security Notes

**This bot runs on YOUR hardware** - please be aware:
- Only whitelist trusted users - they'll be using your GPU/CPU resources
- Malicious or excessive use could overload your system
- For server channels, enable Discord's slow mode to prevent spam
- The whitelist is intentionally config-only (no Discord commands) so only the person running the LLM controls access

## 📁 Managing Multiple Configurations

You can create multiple bot configurations with different settings:

- Each config is saved in the `configs/` folder
- Name configs descriptively (e.g., 'gaming-bot', 'work-assistant')
- Switch between configs when starting the bot
- Edit configs manually by opening the JSON files in any text editor
- After editing, stop the bot (Ctrl+C) and restart it to reload changes

## 🆘 Troubleshooting

If you encounter issues, check the [Troubleshooting Guide](TROUBLESHOOTING.md) for common problems and solutions.
