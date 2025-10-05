# Troubleshooting Guide

## 🔴 Bot Not Working?

### The bot is offline in Discord

**Your PC needs to stay on!** The bot only works when:
- Your computer is on
- The terminal/command window is running `python main.py`
- Your LLM (Ollama/LM Studio) is running

**Quick fix:** Go to your PC and restart the bot:
```bash
python main.py
# Choose [1] to use saved settings
```

---

### "Cannot connect to Ollama/LM Studio"

**For Ollama:**
```bash
# In a new terminal, start Ollama:
ollama serve

# Check if you have models:
ollama list

# If no models, get one:
ollama pull <model-name>
```

**For LM Studio:**
1. Open LM Studio app
2. Click "Developer" tab (left side)
3. Load a model if needed
4. Click "Start Server"

---

### Slash commands not showing up

This is normal! Discord takes 5-60 minutes to show new commands.

**Try this:**
1. Type `/` and wait a second
2. On mobile: Pull down to refresh Discord
3. On desktop: Press Ctrl+R
4. Be patient - they will appear!

---

### Bot responds too slowly

**Use a smaller, faster model:**

For Ollama:
- Browse the [Ollama model library](https://ollama.com/search) for smaller models
- Look for models with lower parameters (1B, 3B are faster than 7B, 13B)
- Pull your chosen model:
```bash
ollama pull <model-name>
```

For LM Studio:
- Look for models with "Q4" 
- Try 1B or 3B parameter models

---

### "Invalid Discord Token"

You need a new token:
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Select your application
3. Go to "Bot" section
4. Click "Reset Token"
5. Copy the new token
6. Run `python main.py` and choose [2] to setup again

---

### Can't DM the bot

**You must add the bot to a server first!**
1. Use the invite URL to add bot to any server
2. Find the bot in the server member list
3. Right-click the bot
4. Open full sized profile
5. Select message icon to start a DM conversation 

---

## 🖼️ Vision Model Issues

### Images not working?

**Check if you're using a vision-capable model:**
- The bot will show "🖼️ Vision model detected!" on startup if your model supports images
- Use `/status` command to check vision support
- Vision models usually have "vision", "llava", or similar in their name

**For Ollama:**
```bash
# Get a vision model
ollama pull <vision-model-name>
```

**For LM Studio:**
- Download a vision-capable model from the Discover tab
- Look for models with vision support in the description

---

### Bot doesn't respond to images

**Make sure:**
1. You're using a vision model (check with `/status`)
2. The image is attached properly (not a link)
3. Discord file size limits (8MB for free servers, 50MB for boosted)
4. Supported formats: JPG, PNG, GIF, WEBP

---

### Slow response with images

**This is normal for vision models:**
- Image processing takes much longer than text
- Larger images = slower processing
- First image after startup may be extra slow (model loading)
- The bot has a 90-second timeout for image processing

**Tips:**
- Use smaller/compressed images when possible
- Be patient - vision models need time
- Consider using a faster vision model if available

---

## 🧠 Reasoning Model Issues

**Important:** The reasoning still happens - models think through problems internally. The bot just hides the thinking process from Discord to keep responses clean. You can see the full chain-of-thought reasoning in your LM Studio or Ollama server logs if you want to observe it.

The bot automatically strips most common thinking tags:
- `<think>...</think>`
- `<thinking>...</thinking>`
- `<reason>...</reason>`
- `<reasoning>...</reasoning>`
- `<thought>...</thought>`
- `<internal>...</internal>`
- `<scratch>...</scratch>`

**If you still see thinking output:**
1. Check your LM Studio server logs to identify the exact tag format
2. [Open a GitHub issue](https://github.com/ella0333/Local-LLM-Discord-Bot/issues) with:
   - Model name you're using
   - The exact text output including thinking tags
   - Which provider (Ollama/LM Studio)
   - Screenshot if possible

**For Ollama users:**
- The bot lets models think naturally (reasoning still happens internally)
- Thinking tags are stripped from Discord output only
- You can see the full reasoning process in Ollama server logs
- Check terminal where `ollama serve` is running to see thinking
- Some older Ollama versions may format thinking differently
- Update Ollama to the latest version: Visit [ollama.com](https://ollama.com)

**For LM Studio users:**
- The bot checks for separate `reasoning_content` field (works automatically)
- Thinking tags in main content are stripped automatically
- You can see the full reasoning in LM Studio's server logs/output
- Check the Developer tab console to see thinking
- Some models have a separate `reasoning_content` field (works automatically)
- Others put thinking in `<think>` tags (automatically stripped)
- Rarely, models may use non-standard tags

**Temporary workaround:**
- Use the model's non-thinking variant if available

**Note**
- Some smaller models with reasoning may spawn thinking without proper formatting or randomly within messages, this is a model issue. 

---

## 💡 Common Questions

### Can I chat when I'm not home?

**Yes!** As long as:
- Your home PC stays on
- The bot is running (`python main.py`)
- Your home internet is working

### Can friends use my bot?

Yes, if you add them to the whitelist during setup and add the bot to a server they're in. Consider enabling moderation for safety!

### How do I stop the bot?

Press `Ctrl+C` in the terminal where it's running.

### How do I change models?

Run `python main.py` and choose [2] to run setup again.

### The bot forgot our conversation

Use `/clear` to intentionally clear history. Otherwise, it remembers the last 10 messages or the amount of messages you have set in the setup. Some smaller models are not trained to handle large context sizes or multi-turn conversations so keep this in mind.

---

## 🆘 Still Need Help?

[Open a GitHub issue](https://github.com/ella0333/Local-LLM-Discord-Bot/issues):
- Technical support
- Bug reports
- Feature requests
- General questions
- **Reasoning model tag issues** (include LM Studio server logs)

Remember: Your computer is doing all the LLM hard lifting, Discord is just the messenger!
