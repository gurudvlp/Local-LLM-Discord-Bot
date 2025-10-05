"""
Utility functions for Discord Local LLM Bot
"""

import sys
import os
import platform
from datetime import datetime
from typing import Optional


def print_banner():
    """Print welcome banner"""
    print(Colors.cyan("""
    ═══════════════════════════════════════════════════════════════
    
                     📱 Discord Local LLM Bot 🤖
    
       Turn Ollama or LM Studio into a Discord bot for mobile 
                      access to your Local LLM.
    
    ═══════════════════════════════════════════════════════════════
    """))


def validate_environment() -> bool:
    """Validate Python version is 3.8 or newer"""
    
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(Colors.red(f"❌ You need Python 3.8 or newer"))
        print(f"   You have: Python {python_version.major}.{python_version.minor}")
        print("\nPlease update Python and try again.")
        return False
    
    return True


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def sanitize_message(text: str) -> str:
    """Remove null characters and enforce Discord's 2000 character limit"""
    if not text:
        return ""
    
    text = text.replace('\x00', '')
    
    if len(text) > 2000:
        text = text[:1997] + "..."
    
    return text


def format_timestamp(timestamp: Optional[str] = None) -> str:
    """Format timestamp as 12-hour time (e.g., 02:30 PM)"""
    if timestamp:
        dt = datetime.fromisoformat(timestamp)
    else:
        dt = datetime.now()
    
    return dt.strftime("%I:%M %p")


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 characters per token"""
    return len(text) // 4


def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Create ASCII progress bar"""
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = int(width * current / total)
    return "[" + "=" * progress + "-" * (width - progress) + "]"


class Colors:
    """ANSI color codes for terminal output"""
    
    # Enable ANSI colors on Windows
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.OKGREEN}{text}{cls.ENDC}"
    
    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls.FAIL}{text}{cls.ENDC}"
    
    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls.WARNING}{text}{cls.ENDC}"
    
    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls.OKBLUE}{text}{cls.ENDC}"
    
    @classmethod
    def cyan(cls, text: str) -> str:
        return f"{cls.OKCYAN}{text}{cls.ENDC}"
    
    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.ENDC}"
    
    @classmethod
    def disable(cls):
        """Disable colors for piped output"""
        cls.HEADER = ''
        cls.OKBLUE = ''
        cls.OKCYAN = ''
        cls.OKGREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''


# Disable colors if output is being piped
if not sys.stdout.isatty():
    Colors.disable()
