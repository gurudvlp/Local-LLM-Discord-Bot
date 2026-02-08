# Message Queue Configuration

## Overview

The bot now includes a message queue system with bot storm protection to prevent message loss and infinite bot conversation loops.

## New Configuration Parameters

Add these optional parameters to your bot configuration JSON file (in `configs/your_config.json`):

### Queue Settings

```json
{
  "queue_max_size": 50,
  "queue_message_ttl": 300
}
```

- **`queue_max_size`** (default: `50`)
  - Maximum number of messages that can be queued per channel
  - When full, bot messages are dropped to make room for human messages

- **`queue_message_ttl`** (default: `300`)
  - Time To Live for queued messages in seconds (5 minutes default)
  - Messages older than this are automatically removed from the queue

### Bot Storm Protection

```json
{
  "enable_bot_storm_protection": true,
  "bot_cooldown_seconds": 3,
  "bot_max_messages_per_minute": 10,
  "consecutive_bot_threshold": 3
}
```

- **`enable_bot_storm_protection`** (default: `true`)
  - Enable/disable bot storm protection
  - When disabled, bot responds to all messages without rate limiting

- **`bot_cooldown_seconds`** (default: `3`)
  - Minimum seconds between bot responses in server channels
  - DMs are not affected by this cooldown
  - Prevents rapid-fire responses

- **`bot_max_messages_per_minute`** (default: `10`)
  - Maximum number of messages the bot can send per minute per channel
  - Uses a sliding 60-second window
  - DMs are not affected by this limit

- **`consecutive_bot_threshold`** (default: `3`)
  - Number of consecutive bot messages that triggers storm detection
  - When detected, bot STOPS responding to bot messages until a human speaks
  - Example: If 3 consecutive messages are from bots, the bot enters "storm mode"

## How It Works

### Priority System

Messages are processed with priority:
- **Human messages**: Priority 10 (HIGH) - Always processed first
- **Bot messages (no storm)**: Priority 5 (MEDIUM) - Processed normally
- **Bot messages (during storm)**: Skipped entirely until a human speaks

### Queue Overflow Behavior

When the queue is full:
1. If new message is from a **human** and queue has **bot messages**: Drop oldest bot message to make room
2. If queue is full of **human messages**: Reject new message with ⏳ reaction
3. If new message is from a **bot**: Reject with ⏳ reaction

### DM Behavior

Direct messages (DMs) are **not affected** by storm protection:
- No cooldowns
- No rate limits
- No storm detection
- Full responsiveness maintained

This ensures 1-on-1 conversations remain uninterrupted.

## Example Configuration

Full example of a config file with queue settings:

```json
{
  "llm_provider": "ollama",
  "llm_base_url": "http://localhost:11434",
  "model_name": "llama3.1:latest",
  "discord_token": "your_discord_token_here",
  "dm_whitelist": ["YourUsername"],
  "server_whitelist": ["YourUsername"],
  "owner_username": "YourUsername",
  "context_window_size": 10,
  "enable_streaming": true,

  "queue_max_size": 50,
  "queue_message_ttl": 300,
  "enable_bot_storm_protection": true,
  "bot_cooldown_seconds": 3,
  "bot_max_messages_per_minute": 10,
  "consecutive_bot_threshold": 3
}
```

## Testing the Queue System

To verify the queue system is working:

1. **Basic Queue**: Send multiple messages rapidly → All should be processed in order
2. **Priority**: Have a bot and human send messages → Human messages processed first
3. **Storm Detection**: Have 3+ bots send consecutive messages → Bot stops responding to bots
4. **Storm Recovery**: After storm, send a human message → Bot resumes normal operation
5. **DM Test**: In DMs, verify no cooldowns or storm protection applies

## Troubleshooting

### Bot not responding in server channels

- Check if storm protection triggered (look for log: "Bot storm detected")
- Send a human message to reset storm mode
- Check if cooldown period is active

### Messages getting ⏳ reaction

- Queue is full or bot is rate-limited
- Wait a few seconds and try again
- If persistent, check `queue_max_size` setting

### Bot responding too slowly

- Increase `queue_max_size` if many users are active
- Decrease `bot_cooldown_seconds` for faster responses
- Increase `bot_max_messages_per_minute` if needed

## Logs

The bot logs queue activity:
- Queue worker starts/stops
- Storm detection triggers
- Queue overflow events
- Stale message removal

Check console output for detailed queue status.
