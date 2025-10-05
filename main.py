"""
Discord Local LLM Bot - Chat with your local AI through Discord
"""
import discord
import asyncio
import sys
import os
import signal
from pathlib import Path
from config_manager import ConfigManager
from bot import DiscordLLMBot
from utils import print_banner, validate_environment, Colors, clear_screen


async def run_bot(config):
    """Run the bot with proper cleanup"""
    bot = DiscordLLMBot(config)
    
    try:
        await bot.start()
    except discord.errors.LoginFailure:
        print(Colors.red("\n❌ Discord token invalid!"))
        print("\nPlease check your token and run setup again.")
        config_name = config.get('_config_name', 'unknown')
        print(f"Config file: configs/{config_name}_config.json")
        await bot.shutdown()
        return False
    except asyncio.CancelledError:
        print(Colors.yellow("\n\nStopping bot..."))
        await bot.shutdown()
        return False
    except Exception as e:
        print(Colors.red(f"\n❌ Error: {e}"))
        await bot.shutdown()
        return False
    
    return False


def main():
    """Main entry point"""
    clear_screen()
    print_banner()
    
    if not validate_environment():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    config_manager = ConfigManager()
    
    if config_manager.config_exists():
        configs = config_manager.list_configs()
        print(Colors.green(f"✅ Found {len(configs)} saved configuration(s)\n"))
        
        print("What would you like to do?\n")
        print("  [1] Load existing configuration")
        print("  [2] Create new configuration")
        print("  [3] Edit configuration (manually)")
        print("  [4] Delete configuration")
        print("  [5] Exit\n")
        
        while True:
            choice = input("Select (1-5): ").strip()
            
            if choice == "1":
                clear_screen()
                print_banner()
                print(Colors.green("\n🚀 Loading configuration...\n"))
                
                config_name = config_manager.choose_config()
                if config_name:
                    config = config_manager.load_config(config_name)
                    if config:
                        print(Colors.green(f"\n✅ Loaded config: '{config_name}'\n"))
                        break
                    else:
                        print(Colors.red(f"❌ Failed to load config '{config_name}'"))
                        print("Please try another option.\n")
                        continue
                else:
                    print(Colors.yellow("\nNo config selected. Please try again.\n"))
                    continue
                
            elif choice == "2":
                clear_screen()
                print_banner()
                config = config_manager.setup_wizard()
                if not config:
                    print("\nSetup cancelled.")
                    input("\nPress Enter to exit...")
                    sys.exit(0)
                break
                
            elif choice == "3":
                print(Colors.cyan("\n📝 To edit a configuration:"))
                print(Colors.cyan("  1. Navigate to the 'configs' folder"))
                print(Colors.cyan("  2. Open any *_config.json file in a text editor"))
                print(Colors.cyan("  3. Edit the values (keep the JSON format valid!)"))
                print(Colors.cyan("  4. Save the file"))
                print(Colors.cyan("  5. Stop the bot (Ctrl+C) and restart it to reload settings\n"))
                
                configs = config_manager.list_configs()
                if configs:
                    print(Colors.yellow("Available configs:"))
                    for cfg in configs:
                        print(f"  • configs/{cfg}_config.json")
                
                print()
                input("Press Enter to go back...")
                clear_screen()
                print_banner()
                print(Colors.green(f"✅ Found {len(configs)} saved configuration(s)\n"))
                print("What would you like to do?\n")
                print("  [1] Load existing configuration")
                print("  [2] Create new configuration")
                print("  [3] Edit configuration (manually)")
                print("  [4] Delete configuration")
                print("  [5] Exit\n")
                continue
                
            elif choice == "4":
                print(Colors.yellow("\n⚠️  Delete configuration\n"))
                config_name = config_manager.choose_config()
                if config_name:
                    confirm = input(f"\nAre you sure you want to delete '{config_name}'? (y/n): ").strip().lower()
                    if confirm == 'y':
                        if config_manager.delete_config(config_name):
                            print(Colors.green(f"✅ Deleted config '{config_name}'"))
                        else:
                            print(Colors.red(f"❌ Failed to delete config '{config_name}'"))
                    else:
                        print("Deletion cancelled.")
                
                if not config_manager.config_exists():
                    print(Colors.yellow("\n⚠️  No configurations remaining."))
                    print("Please create a new configuration.\n")
                    input("Press Enter to continue...")
                    clear_screen()
                    print_banner()
                    config = config_manager.setup_wizard()
                    if not config:
                        print("\nSetup cancelled.")
                        input("\nPress Enter to exit...")
                        sys.exit(0)
                    break
                else:
                    print()
                    input("Press Enter to go back...")
                    clear_screen()
                    print_banner()
                    configs = config_manager.list_configs()
                    print(Colors.green(f"✅ Found {len(configs)} saved configuration(s)\n"))
                    print("What would you like to do?\n")
                    print("  [1] Load existing configuration")
                    print("  [2] Create new configuration")
                    print("  [3] Edit configuration (manually)")
                    print("  [4] Delete configuration")
                    print("  [5] Exit\n")
                    continue
                
            elif choice == "5":
                print("\nGoodbye!")
                sys.exit(0)
            else:
                print(Colors.yellow("Please enter 1, 2, 3, 4, or 5"))
    else:
        print(Colors.yellow("No configurations found\n"))
        print("Let's create your first bot configuration!\n")
        input("Press Enter to continue...")
        clear_screen()
        print_banner()
        
        config = config_manager.setup_wizard()
        if not config:
            print("\nSetup cancelled.")
            input("\nPress Enter to exit...")
            sys.exit(0)
    
    config_name = config.get('_config_name', 'unknown')
    print(Colors.blue("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    print(Colors.green(f"    🤖 Starting Discord Bot"))
    print(Colors.cyan(f"    Config: '{config_name}'"))
    print(Colors.blue("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"))
    
    # Handle graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    main_task = loop.create_task(run_bot(config))
    
    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        print(Colors.yellow("\n\nReceived interrupt signal..."))
        main_task.cancel()
        try:
            loop.run_until_complete(main_task)
        except asyncio.CancelledError:
            pass
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()
    
    print(Colors.green("\n✅ Bot stopped"))
    print("Run 'python main.py' to start again.")
    print(Colors.cyan("\n💡 Note: You can stop your LLM server now if finished"))
    print(Colors.cyan("   - Ollama: Press Ctrl+C in the ollama serve terminal"))
    print(Colors.cyan("   - LM Studio: Click 'Stop Server' in Developer tab"))


if __name__ == "__main__":
    try:
        import discord
    except ImportError:
        print(Colors.red("❌ Discord.py not installed!"))
        print("\nRun: pip install -r requirements.txt")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    main()
