import copy
import re
from typing import List, Optional
import zipfile
from bs4 import BeautifulSoup, NavigableString


def load_epub_html(epub_path: str, doc_range: Optional[tuple[int, int]] = None) -> List[str]:
    """Load the HTML index files from an epub and return a list of HTML strings, one for each index file.
    
    Args:
        epub_path: Path to epub file
        doc_range: Tuple specifying the index range of html docs to load (start, end)        
    Returns:
        List of HTML strings
    """

    # Default to loading first FILE_NO docs if range not specified
    if doc_range is None:
        doc_range = (1, 10)
    
    start_index, end_index = doc_range

    html_contents = []
    
    with zipfile.ZipFile(epub_path, 'r') as epub:
        # List all files in epub
        files = epub.namelist()
        
        # Find HTML files (typically in OEBPS/text/)
        html_files = [f for f in files if f.endswith('.html') or f.endswith('.xhtml')]
        html_files.sort()  # Ensure files are in order
        
        # Load HTML files within specified range
        for i in range(start_index - 1, min(end_index, len(html_files))):
            with epub.open(html_files[i]) as f:
                content = f.read().decode('utf-8')
                
                # Clean up content - remove doctype, xml declarations
                content = re.sub(r'<!DOCTYPE[^>]*>', '', content)
                content = re.sub(r'<\?xml[^>]*\?>', '', content)
                
                html_contents.append(content)
                
    return html_contents

def get_id(tag):
    return f"{tag.name}_{id(tag)}"

def get_ids(tags):
    return [get_id(tag) for tag in tags]

def extract_tagged_blocks(html_text: str, start_index: int, end_index: Optional[int] = None) -> str:
    soup = BeautifulSoup(html_text, 'html.parser')
    
    # Get target tags
    all_tags = [tag for tag in soup.find_all(['h1','h2','h3','p']) 
                if not isinstance(tag, NavigableString)]

    # Validate indices
    if start_index < 1:
        raise ValueError("start_index must be >= 1")
    if end_index and end_index < start_index:
        raise ValueError("end_index must be >= start_index")

    start_idx = start_index - 1
    end_idx = end_index - 1 if end_index else len(all_tags) - 1

    if start_idx >= len(all_tags):
        raise ValueError(f"start_index {start_index} exceeds number of tags ({len(all_tags)})")
    if end_idx >= len(all_tags):
        end_idx = len(all_tags) - 1

    # Get selected tags and their ancestors
    selected_tags = all_tags[start_idx:end_idx + 1]
    required_tags = set()

    # Collect all ancestors
    for tag in selected_tags:
        required_tags.add(tag)
        for parent in tag.parents:
            required_tags.add(parent)

    # Create new document
    new_soup = BeautifulSoup('', 'html.parser')

    # Function to recursively build minimal tree
    def build_minimal_tree(tag):
        if get_id(tag) in get_ids(required_tags):
            new_tag = copy.copy(tag)
            if not new_tag in selected_tags:
                new_tag.clear()  # Remove all children
            else:
                return new_tag
            
            # Only add required children
            for child in tag.children:
                if not isinstance(child, NavigableString) and get_id(child) in get_ids(required_tags):
                    new_tag.append(build_minimal_tree(child))
            return new_tag
        return None

    # Start from root of selected tags
    root = list(required_tags)[0]
    while root.parent in required_tags:
        root = root.parent

    new_soup.append(build_minimal_tree(root))
    return str(new_soup)


from typing import Any, Dict, List, Tuple
import json
from bs4 import Tag
from colorama import Fore, Style, init

# Initialize colorama
init()

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

def is_complete_sentence(text):
    # Define regex patterns for various sentence-ending formats
    sentence_end_patterns = [
        r'[.!?][\"\')\]]*$',  # Ends with punctuation followed by optional quotes, parentheses, or brackets
        r'[.!?][\"\')\]]*\s*\(\d+\)$',  # Ends with punctuation followed by optional quotes, parentheses, or brackets and a citation
        r'[.!?][\"\')\]]*\s*\[\d+\]$',  # Ends with punctuation followed by optional quotes, parentheses, or brackets and a footnote
        r'[.!?][\"\')\]]*$',  # Ends with punctuation followed by optional quotes, parentheses, or brackets and space
        r':$',  # Ends with colon (lead up to a block quotation)
    ]

    # Check if the text matches any of the sentence-ending patterns
    for pattern in sentence_end_patterns:
        if re.search(pattern, text.strip()):
            return True

    return False

def is_ignored(paragraph, ignore_patterns: list[str]):
    for pattern in ignore_patterns:
        if re.search(pattern, paragraph):
            return True
    return False

    
def _clean_paragraphs(paragraphs: list[str], ignore_patterns=[r'^\d+$', r'^\x0c']):
    cleaned_paragraphs = []
    previous_paragraph_incomplete = False
    for paragraph in paragraphs:
        if not is_ignored(paragraph, ignore_patterns):
            lines = paragraph.splitlines()
            if previous_paragraph_incomplete:
                cleaned_paragraphs[-1] += (paragraph)
            else:
                cleaned_paragraphs.append(paragraph)
            previous_paragraph_incomplete = not is_complete_sentence([-1])
    return cleaned_paragraphs