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

def normalize_quotes(s: str | None) -> str:
    """
    Convert all types of single/double quotes into a standard form and remove them.
    You can modify to keep them if you prefer, e.g. replacing them with a single symbol
    instead of removing.
    """
    if s is None:
        return ""
    
    # Replace fancy single quotes
    s = s.replace("‘", "'").replace("’", "'")
    # Replace fancy double quotes
    s = s.replace("“", '"').replace("”", '"')

    return s

def clean_string(x: str) -> str:
    if x is None:
        return ""
    x = normalize_quotes(x)
    return " ".join(x.split())

def strings_equal(s1: str, s2: str, case_sensitive: bool = True) -> bool:
    """
    Compare two strings after normalizing/removing any kind of
    single/double/fancy quotes. Also normalizes whitespace.
    """
    # Normalize quotes and collapse whitespace
    if not (s1 or s2):
        return False
    
    s1_clean = clean_string(s1)
    s2_clean = clean_string(s2)

    if not case_sensitive:
        s1_clean = s1_clean.lower()
        s2_clean = s2_clean.lower()
    
    return s1_clean == s2_clean
    
def clean_paragraphs(paragraphs: list[str], ignore_patterns=[r'^\d+$', r'^\x0c']):
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