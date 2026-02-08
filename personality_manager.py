"""
Personality Manager - Manages custom personality slots per channel
Bot-specific storage using composite keys (bot_id, guild_id, channel_id)
"""

import json
from pathlib import Path
from typing import Dict, Optional, List


class PersonalityManager:
    """Manages personality instructions for Discord channels with bot-specific storage"""

    PERSONALITIES_FILE = Path("personalities.json")
    MAX_SLOTS = 5

    def __init__(self):
        """Initialize personality manager and load existing personalities"""
        self.personalities: Dict[str, Dict[str, str]] = {}  # key format: "botid_guildid_channelid"
        self._load_personalities()

    @staticmethod
    def _make_key(bot_id: int, guild_id: Optional[int], channel_id: int) -> str:
        """
        Create composite key for bot-specific storage.
        Format: "botid_guildid_channelid"
        For DM channels, guild_id is None and stored as "dm"
        """
        guild = guild_id if guild_id is not None else "dm"
        return f"{bot_id}_{guild}_{channel_id}"

    def _load_personalities(self) -> None:
        """Load personalities from JSON file"""
        if self.PERSONALITIES_FILE.exists():
            try:
                with open(self.PERSONALITIES_FILE, 'r') as f:
                    data = json.load(f)
                    # New format uses string keys like "botid_guildid_channelid"
                    # Old format used integer keys (channel IDs only)
                    # Keep string keys as-is (new format)
                    self.personalities = {str(k): v for k, v in data.items()}
            except Exception as e:
                print(f"⚠️  Couldn't load personalities: {e}")
                self.personalities = {}

    def _save_personalities(self) -> bool:
        """Save personalities to JSON file"""
        try:
            with open(self.PERSONALITIES_FILE, 'w') as f:
                # Personalities already use string keys in new format
                json.dump(self.personalities, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Couldn't save personalities: {e}")
            return False

    def set_personality(self, bot_id: int, guild_id: Optional[int], channel_id: int, slot: int, text: str) -> tuple[bool, str]:
        """
        Set a personality instruction for a channel slot.
        Args:
            bot_id: Discord bot's user ID
            guild_id: Discord server/guild ID (None for DMs)
            channel_id: Discord channel ID
            slot: Personality slot number (0-4)
            text: Personality instruction text
        Returns (success, message)
        """
        if not 0 <= slot < self.MAX_SLOTS:
            return False, f"❌ Slot must be 0-{self.MAX_SLOTS - 1}"

        if not text.strip():
            return False, "❌ Personality text cannot be empty"

        key = self._make_key(bot_id, guild_id, channel_id)

        if key not in self.personalities:
            self.personalities[key] = {}

        self.personalities[key][str(slot)] = text.strip()

        if self._save_personalities():
            return True, f"✅ Personality slot {slot} set!"
        else:
            return False, "❌ Failed to save personality"

    def get_personality(self, bot_id: int, guild_id: Optional[int], channel_id: int, slot: int) -> Optional[str]:
        """Get a specific personality instruction"""
        key = self._make_key(bot_id, guild_id, channel_id)

        if key not in self.personalities:
            return None

        return self.personalities[key].get(str(slot))

    def get_personalities(self, bot_id: int, guild_id: Optional[int], channel_id: int) -> List[str]:
        """
        Get all non-empty personality instructions for a channel.
        Args:
            bot_id: Discord bot's user ID
            guild_id: Discord server/guild ID (None for DMs)
            channel_id: Discord channel ID
        Returns list of personality strings in slot order.
        """
        key = self._make_key(bot_id, guild_id, channel_id)

        if key not in self.personalities:
            return []

        slots = self.personalities[key]
        personalities = []

        for i in range(self.MAX_SLOTS):
            personality = slots.get(str(i))
            if personality:
                personalities.append(personality)

        return personalities

    def clear_personality(self, bot_id: int, guild_id: Optional[int], channel_id: int, slot: int) -> tuple[bool, str]:
        """
        Clear a personality instruction from a slot.
        Returns (success, message)
        """
        if not 0 <= slot < self.MAX_SLOTS:
            return False, f"❌ Slot must be 0-{self.MAX_SLOTS - 1}"

        key = self._make_key(bot_id, guild_id, channel_id)

        if key not in self.personalities:
            return False, f"❌ No personalities set for this channel"

        if str(slot) not in self.personalities[key]:
            return False, f"❌ Slot {slot} is already empty"

        del self.personalities[key][str(slot)]

        # Clean up empty entries
        if not self.personalities[key]:
            del self.personalities[key]

        if self._save_personalities():
            return True, f"✅ Cleared personality slot {slot}!"
        else:
            return False, "❌ Failed to save personalities"

    def clear_all_personalities(self, bot_id: int, guild_id: Optional[int], channel_id: int) -> tuple[bool, str]:
        """
        Clear all personality instructions for a channel.
        Returns (success, message)
        """
        key = self._make_key(bot_id, guild_id, channel_id)

        if key not in self.personalities:
            return False, "❌ No personalities set for this channel"

        del self.personalities[key]

        if self._save_personalities():
            return True, "✅ Cleared all personalities for this channel!"
        else:
            return False, "❌ Failed to save personalities"

    def list_personalities(self, bot_id: int, guild_id: Optional[int], channel_id: int) -> str:
        """Get a formatted string of all personalities for a channel"""
        key = self._make_key(bot_id, guild_id, channel_id)

        if key not in self.personalities or not self.personalities[key]:
            return "No personalities set for this channel"

        slots = self.personalities[key]
        lines = []

        for i in range(self.MAX_SLOTS):
            personality = slots.get(str(i))
            if personality:
                # Truncate long personalities for display
                display_text = personality[:100] + "..." if len(personality) > 100 else personality
                lines.append(f"  Slot {i}: {display_text}")

        return "\n".join(lines) if lines else "No personalities set"
