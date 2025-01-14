import logging
from pathlib import Path
import traceback

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

def setup_logging():
    """Configure logging"""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        filename=log_dir / "viewer.log",
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("epubkit.viewer")

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