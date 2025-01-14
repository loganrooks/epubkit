import datetime
import logging
from pathlib import Path
import traceback
from venv import logger

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