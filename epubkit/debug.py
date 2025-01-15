from datetime import datetime
import json
import logging
import os
from pathlib import Path
import traceback
from typing import Any, Dict, List, NamedTuple, Tuple
from venv import logger
from bs4 import Tag
from colorama import Fore, Style, init
from functools import wraps
from typing import Any, Callable, Dict, List
import inspect
import pytest
from dataclasses import dataclass, field
from datetime import datetime


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
    """Get name of class where function is called"""
    frame = inspect.currentframe()
    try:
        # Walk up call stack to find first non-debug.py frame
        while frame:
            if frame.f_code.co_filename != __file__:
                # Check frame locals for class instance
                for obj in frame.f_locals.values():
                    if hasattr(obj, '__class__'):
                        return obj.__class__.__name__
            frame = frame.f_back
    finally:
        del frame  # Avoid reference cycles
    return ''

def get_file_name():
    """Get name of file where function is called"""
    frame = inspect.currentframe()
    try:
        # Walk up call stack to find first non-debug.py frame
        while frame:
            if frame.f_code.co_filename != __file__:
                # Get just the filename without path or extension
                return os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]
            frame = frame.f_back
    finally:
        del frame
    return ''

def get_log_name():
    """Generate log name from caller's class and file"""
    class_name = get_class_name()
    file_name = get_file_name()
    # Combine class and file names, filtering out empty strings
    parts = [p for p in (class_name, file_name) if p]
    return '_'.join(parts)



def setup_logging():
    """Configure logging for caller's context"""
    # Get caller context
    context = get_caller_context()
    module_name = f"epubkit.{context.file_name}"
    if context.class_name:
        module_name += f".{context.class_name}"
    
    # Create logs directory in workspace root
    workspace_root = Path(__file__).parent.parent
    log_dir = workspace_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging with caller-specific file
    log_path = log_dir / f"{context.file_name}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(module_name)

class CallerContext(NamedTuple):
    function_name: str
    class_name: str
    file_name: str

def get_caller_context() -> CallerContext:
    """Get the context where function was called"""
    frame = inspect.currentframe()
    try:
        # Walk up call stack to find first non-debug.py frame
        while frame:
            if frame.f_code.co_filename != __file__:
                # Get function name
                func_name = frame.f_code.co_name
                
                # Get class name if exists
                class_name = ''
                for obj in frame.f_locals.values():
                    if hasattr(obj, '__class__'):
                        class_name = obj.__class__.__name__
                        break
                
                # Get filename without path/extension
                file_name = Path(frame.f_code.co_filename).stem
                
                return CallerContext(func_name, class_name, file_name)
            frame = frame.f_back
    finally:
        del frame  # Avoid reference cycles
    
    return CallerContext('', '', '')

def log_error(error: Exception):
        """Handle errors with filtered traceback and detailed logging"""
        tb = error.__traceback__
        filtered_tb = filter_traceback(tb)
        context = get_caller_context()
            
        logger.error(
            "Error occurred\n"
            f"Error Type: {type(error).__name__}\n"
            f"Error Message: {str(error)}\n"
            f"Time: {datetime.now().isoformat()}\n"
            f"Function: {context.function_name}\n"
            f"Class: {context.class_name}\n"
            f"File: {context.file_name}\n"
            f"Traceback:\n{filtered_tb}\n"
            f"Full traceback:\n{''.join(traceback.format_tb(tb))}"
        )
        

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




@dataclass
class ValueTracker:
    name: str
    value_history: List[tuple[Any, str, datetime]] = field(default_factory=list)
    
    def track(self, value: Any, location: str):
        self.value_history.append((value, location, datetime.now()))
        
    def get_history(self) -> str:
        return "\n".join(
            f"Value: {v}, Location: {loc}, Time: {t}" 
            for v, loc, t in self.value_history
        )

def debug_test(watched_vars: List[str] = None):
    """Decorator to add debugging info to tests"""
    def decorator(test_func: Callable):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            # Initialize value trackers
            trackers = {
                name: ValueTracker(name)
                for name in (watched_vars or [])
            }
            
            # Create tracking function in closure
            def _track(name: str, value: Any) -> Any:
                if name in trackers:
                    frame = inspect.currentframe()
                    location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
                    trackers[name].track(value, location)
                    del frame
                return value
            
            try:
                # Make _track available to test function
                test_globals = test_func.__globals__
                test_globals['_track'] = _track
                
                # Run test
                result = test_func(*args, **kwargs)
                
                return result
                
            except AssertionError as e:
                # Build debug info
                debug_info = [
                    "\n=== Test Debug Info ===",
                    f"Test: {test_func.__name__}",
                    f"Time: {datetime.now()}",
                    "\nValue History:"
                ]
                
                for name, tracker in trackers.items():
                    debug_info.extend([
                        f"\n{name}:",
                        tracker.get_history()
                    ])
                
                debug_info.extend([
                    "\nTest Source:",
                    inspect.getsource(test_func)
                ])
                
                raise AssertionError("\n".join(debug_info)) from e
                
            finally:
                # Clean up
                if '_track' in test_globals:
                    del test_globals['_track']
                    
        return wrapper
    return decorator