"""
Personality Manager - Manages custom personality slots per channel
"""

import json
from pathlib import Path
from typing import Dict, Optional, List


class PersonalityManager:
    """Manages personality instructions for Discord channels"""

    PERSONALITIES_FILE = Path("personalities.json")
    MAX_SLOTS = 5

    def __init__(self):
        """Initialize personality manager and load existing personalities"""
        self.personalities: Dict[int, Dict[str, str]] = {}
        self._load_personalities()

    def _load_personalities(self) -> None:
        """Load personalities from JSON file"""
        if self.PERSONALITIES_FILE.exists():
            try:
                with open(self.PERSONALITIES_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert string keys to integers for channel IDs
                    self.personalities = {int(k): v for k, v in data.items()}
            except Exception as e:
                print(f"⚠️  Couldn't load personalities: {e}")
                self.personalities = {}

    def _save_personalities(self) -> bool:
        """Save personalities to JSON file"""
        try:
            with open(self.PERSONALITIES_FILE, 'w') as f:
                # Convert integer keys to strings for JSON compatibility
                json_data = {str(k): v for k, v in self.personalities.items()}
                json.dump(json_data, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Couldn't save personalities: {e}")
            return False

    def set_personality(self, channel_id: int, slot: int, text: str) -> tuple[bool, str]:
        """
        Set a personality instruction for a channel slot.
        Returns (success, message)
        """
        if not 0 <= slot < self.MAX_SLOTS:
            return False, f"❌ Slot must be 0-{self.MAX_SLOTS - 1}"

        if not text.strip():
            return False, "❌ Personality text cannot be empty"

        if channel_id not in self.personalities:
            self.personalities[channel_id] = {}

        self.personalities[channel_id][str(slot)] = text.strip()

        if self._save_personalities():
            return True, f"✅ Personality slot {slot} set!"
        else:
            return False, "❌ Failed to save personality"

    def get_personality(self, channel_id: int, slot: int) -> Optional[str]:
        """Get a specific personality instruction"""
        if channel_id not in self.personalities:
            return None

        return self.personalities[channel_id].get(str(slot))

    def get_personalities(self, channel_id: int) -> List[str]:
        """
        Get all non-empty personality instructions for a channel.
        Returns list of personality strings in slot order.
        """
        if channel_id not in self.personalities:
            return []

        slots = self.personalities[channel_id]
        personalities = []

        for i in range(self.MAX_SLOTS):
            personality = slots.get(str(i))
            if personality:
                personalities.append(personality)

        return personalities

    def clear_personality(self, channel_id: int, slot: int) -> tuple[bool, str]:
        """
        Clear a personality instruction from a slot.
        Returns (success, message)
        """
        if not 0 <= slot < self.MAX_SLOTS:
            return False, f"❌ Slot must be 0-{self.MAX_SLOTS - 1}"

        if channel_id not in self.personalities:
            return False, f"❌ No personalities set for this channel"

        if str(slot) not in self.personalities[channel_id]:
            return False, f"❌ Slot {slot} is already empty"

        del self.personalities[channel_id][str(slot)]

        # Clean up empty channel entries
        if not self.personalities[channel_id]:
            del self.personalities[channel_id]

        if self._save_personalities():
            return True, f"✅ Cleared personality slot {slot}!"
        else:
            return False, "❌ Failed to save personalities"

    def clear_all_personalities(self, channel_id: int) -> tuple[bool, str]:
        """
        Clear all personality instructions for a channel.
        Returns (success, message)
        """
        if channel_id not in self.personalities:
            return False, "❌ No personalities set for this channel"

        del self.personalities[channel_id]

        if self._save_personalities():
            return True, "✅ Cleared all personalities for this channel!"
        else:
            return False, "❌ Failed to save personalities"

    def list_personalities(self, channel_id: int) -> str:
        """Get a formatted string of all personalities for a channel"""
        if channel_id not in self.personalities or not self.personalities[channel_id]:
            return "No personalities set for this channel"

        slots = self.personalities[channel_id]
        lines = []

        for i in range(self.MAX_SLOTS):
            personality = slots.get(str(i))
            if personality:
                # Truncate long personalities for display
                display_text = personality[:100] + "..." if len(personality) > 100 else personality
                lines.append(f"  Slot {i}: {display_text}")

        return "\n".join(lines) if lines else "No personalities set"
