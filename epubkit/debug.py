import datetime
import json
import logging
from pathlib import Path
import traceback
from typing import Any, Dict, List, Tuple
from venv import logger
from bs4 import Tag
from colorama import Fore, Style, init

# Initialize colorama
init()

init(autoreset=True)


def filter_traceback(tb) -> str:
    """Filter and format traceback to only show our code"""
    filtered_lines = []
    while tb:
        frame = tb.tb_frame
        if "site-packages" not in frame.f_code.co_filename:
            # Format frame info similar to standard traceback
            filename = frame.f_code.co_filename
            function = frame.f_code.co_name
            lineno = tb.tb_lineno
            
            # Get the line of code
            with open(filename) as f:
                code = list(f.readlines())[lineno - 1].strip()
                
            # Format like standard traceback
            frame_fmt = (
                f"  File \"{filename}\", line {lineno}, in {function}\n"
                f"    {code}\n"
            )
            filtered_lines.append(frame_fmt)
            
        tb = tb.tb_next
        
    return "Traceback (most recent call last):\n" + "".join(filtered_lines)

import inspect

def get_class_name():
    # Get the current frame
    frame = inspect.currentframe()
    # Get the caller's frame
    caller_frame = frame.f_back
    # Get the caller's class name
    class_name = ''
    for obj in caller_frame.f_locals.values():
        if isinstance(obj, type):
            class_name = obj.__name__
            break
    return class_name

def get_file_name():
    # Get the current frame
    frame = inspect.currentframe()
    # Get the caller's frame
    caller_frame = frame.f_back
    # Get the file name from the caller's frame
    file_name = caller_frame.f_code.co_filename
    return file_name

def get_log_name():
    class_name = get_class_name()
    file_name = get_file_name()
    return f"{class_name}_{file_name}"

def setup_logging():
    """Configure logging"""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        filename=log_dir / f"{get_log_name()}.log",
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("epubkit.viewer")

def log_error(self, error: Exception):
        """Handle errors with filtered traceback and detailed logging"""
        tb = error.__traceback__
        filtered_tb = filter_traceback(tb)
            
        logger.error(
            "Error occurred\n"
            f"Error Type: {type(error).__name__}\n"
            f"Error Message: {str(error)}\n"
            f"Time: {datetime.now().isoformat()}\n"
            f"Context: {self.__class__.__name__}\n"
            f"Traceback:\n{filtered_tb}\n"
            f"Full traceback:\n{''.join(traceback.format_tb(tb))}"
        )
            

def main():
    try:
        # Your code that may raise an exception
        raise ValueError("An example error")
    except Exception as e:
        # Get the current traceback
        tb = e.__traceback__
        # Filter and format the traceback
        filtered_tb = filter_traceback(tb)
        # Print the filtered traceback
        print(filtered_tb)

if __name__ == "__main__":
    main()


def debug_print_dict(d: Dict, indent: int = 0, title: str = None) -> None:
    """Pretty prints nested dictionary structures with colors"""
    if title:
        print(f"\n{Fore.CYAN}{title}{Style.RESET_ALL}")

    def _print_item(item: Any, depth: int = 0) -> None:
        prefix = "  " * depth

        if isinstance(item, dict):
            for k, v in item.items():
                print(f"{prefix}{Fore.GREEN}{k}{Style.RESET_ALL}:")
                _print_item(v, depth + 1)
        elif isinstance(item, (list, tuple)):
            for i, v in enumerate(item):
                print(f"{prefix}{Fore.BLUE}[{i}]{Style.RESET_ALL}:")
                _print_item(v, depth + 1)
        else:
            print(f"{prefix}{item}")

    _print_item(d)


def debug_print_pattern_key(pattern_key: Tuple, matches: List[Tag]) -> None:
    """Visualize pattern key and its matches"""
    print(f"\n{Fore.YELLOW}Pattern Key Structure:{Style.RESET_ALL}")
    for i, (tag_name, classes) in enumerate(pattern_key):
        print(f"  {i}: {Fore.GREEN}{tag_name}{Style.RESET_ALL} "
              f"[{Fore.BLUE}{', '.join(classes)}{Style.RESET_ALL}]")

    print(f"\n{Fore.YELLOW}Matching Tags ({len(matches)}):{Style.RESET_ALL}")
    for i, tag in enumerate(matches[:3]):  # Show first 3 matches
        print(f"  {i}: {tag.name} - {tag.get_text()[:50]}...")
    if len(matches) > 3:
        print(f"  ... and {len(matches)-3} more matches")


def debug_print_tag_structure(tag: Tag, max_depth: int = 3) -> None:
    """Visualize HTML tag hierarchy"""
    def _print_tag(t: Tag, depth: int = 0) -> None:
        if depth > max_depth:
            return

        prefix = "  " * depth
        classes = ' '.join(t.get('class', []))
        text = t.get_text()[:50].replace('\n', ' ').strip()

        print(f"{prefix}{Fore.GREEN}{t.name}{Style.RESET_ALL} "
              f"[{Fore.BLUE}{classes}{Style.RESET_ALL}] "
              f"→ {text}...")

        for child in t.children:
            if isinstance(child, Tag):
                _print_tag(child, depth + 1)

    print(f"\n{Fore.YELLOW}Tag Structure:{Style.RESET_ALL}")
    _print_tag(tag)


def debug_print_pattern_match(pattern: Dict, tag: Tag) -> None:
    """Visualize pattern matching against a tag"""
    print(f"\n{Fore.YELLOW}Pattern Match Analysis:{Style.RESET_ALL}")
    print(f"Pattern: {json.dumps(pattern, indent=2)}")
    print(f"Tag: {tag.name} [{' '.join(tag.get('class', []))}]")

    matches = []
    mismatches = []

    for key, value in pattern.items():
        actual = getattr(tag, key, None)
        if actual == value:
            matches.append(key)
        else:
            mismatches.append((key, value, actual))

    print(f"\n{Fore.GREEN}Matches:{Style.RESET_ALL}")
    for m in matches:
        print(f"  ✓ {m}")

    print(f"\n{Fore.RED}Mismatches:{Style.RESET_ALL}")
    for key, expected, actual in mismatches:
        print(f"  ✗ {key}:")
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")