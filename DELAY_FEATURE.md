# Artificial Response Delay Feature

## Overview
This feature allows you to add an artificial delay before your bot responds to messages, making fast bots appear to be running on slower hardware. This is perfect for making a Markov Chain bot appear to "think" longer, simulating an older or slower system.

## How It Works

1. **During Setup**: When creating a new bot configuration, you'll be asked if you want to enable artificial delay (Step 13)
2. **Configuration**: You specify a minimum and maximum delay in seconds (e.g., 10-30 or 30-90)
3. **Runtime**: Before responding to each message, the bot will:
   - Generate a random delay between your min and max values
   - Show the "typing..." indicator during this time
   - Then generate and send the actual response

## Setup Instructions

### For New Bots
When running the setup wizard, you'll see:

```
Step 13: Artificial Response Delay (Optional)

Add artificial delay before responding to simulate slower hardware?
• Useful for making a fast bot appear slower
• Bot will show 'typing...' indicator during delay
• Delay is randomized between min and max values

Enable artificial delay? (y/n):
```

If you choose `y`, you'll be prompted for:
- **Minimum delay (seconds)**: The shortest time to wait (e.g., 10)
- **Maximum delay (seconds)**: The longest time to wait (e.g., 30)

### For Existing Bots
To add delay to an existing bot, manually edit your config file:

1. Open `configs/your-bot-name_config.json`
2. Add these lines:
```json
{
  "response_delay_min": 10,
  "response_delay_max": 30,
  ...other settings...
}
```

## Example Use Cases

### Fast API Bot (OpenAI)
```json
"response_delay_min": 0,
"response_delay_max": 0
```
No delay - responds immediately

### Markov Chain Bot (Simulating Old Hardware)
```json
"response_delay_min": 30,
"response_delay_max": 90
```
Waits 30-90 seconds before responding, making it appear to be running on slower hardware

### Moderate Slowdown
```json
"response_delay_min": 5,
"response_delay_max": 15
```
Adds a short 5-15 second delay for a subtle slowdown effect

## Viewing Configuration

Use the `/status` command to see the current delay configuration for a bot:
```
Response Delay: 30-90s
```

When the bot starts, it will also display:
```
⏱️  Artificial Delay: 30-90 seconds (simulating slower hardware)
```

## Technical Details

- The delay is applied **before** the LLM generates the response
- The Discord "typing..." indicator shows during the entire delay period
- Each response gets a fresh random delay within your specified range
- The delay applies to both regular messages and slash commands
- Works with all LLM providers (Ollama, Claude, OpenAI, etc.)

## Example: Two Bots Talking

With two bots configured like this:

**fast-bot** (OpenAI):
```json
"response_delay_min": 0,
"response_delay_max": 0
```

**slow-bot** (Markov):
```json
"response_delay_min": 30,
"response_delay_max": 90
```

The fast-bot will respond instantly while the slow-bot will "think" for 30-90 seconds before each response, creating an interesting dynamic!
