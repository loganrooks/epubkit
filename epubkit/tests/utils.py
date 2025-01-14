from bs4 import BeautifulSoup
from colorama import Fore, Style
from typing import Dict, List, Optional, Set, Tuple

from epubkit.parser import HTMLCategoryExtractor

def analyze_extraction_coverage(
    html_content: str,
    pattern_groups: Dict[Tuple, List[BeautifulSoup]],
    pattern_extractor: Optional[HTMLCategoryExtractor] = None
) -> None:
    """
    Analyze extraction coverage and diagnose missing matches
    
    Args:
        html_content: Original HTML content
        pattern_groups: Extracted pattern groups and matches
        pattern_extractor: Optional extractor to diagnose pattern matching
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    all_text_blocks = []
    
    # Get all text blocks with their tag context
    for text in soup.find_all(text=True):
        if not text.strip():
            continue
        if text and len(text) > 3:
            current = text.parent
            tag_hierarchy = []
            while current and current.name:
                tag_hierarchy.append({
                    'name': current.name,
                    'classes': current.get('class', []),
                    'attrs': {k:v for k,v in current.attrs.items() 
                            if k not in ['class', 'id']}
                })
                current = current.parent
            all_text_blocks.append((text, tag_hierarchy))
    
    # Get extracted text
    extracted_text = set()
    for matches in pattern_groups.values():
        for tag in matches:
            text = tag.get_text().strip()
            if text:
                extracted_text.add(text)
    
    # Find missing text and analyze
    missing_blocks = [(text, hierarchy) 
                     for text, hierarchy in all_text_blocks 
                     if text not in extracted_text]
    
    # Print statistics
    total = len(all_text_blocks)
    extracted = len(extracted_text)
    missing = len(missing_blocks)
    
    print(f"\n{Fore.CYAN}=== Extraction Coverage Analysis ==={Style.RESET_ALL}")
    print(f"Total text blocks: {total}")
    print(f"Extracted: {Fore.GREEN}{extracted} ({extracted/total*100:.1f}%){Style.RESET_ALL}")
    print(f"Missing: {Fore.RED}{missing} ({missing/total*100:.1f}%){Style.RESET_ALL}")

    # Print extracted samples
    print(f"\n{Fore.GREEN}=== Sample Extracted Text ==={Style.RESET_ALL}")
    for text in list(extracted_text)[:5]:  # Show first 5
        print(f"✓ {text[:100]}...")
    if len(extracted_text) > 5:
        print(f"... and {len(extracted_text)-5} more")
    
    if missing_blocks:
        print(f"\n{Fore.RED}=== Missing Text Analysis ==={Style.RESET_ALL}")
        for text, hierarchy in missing_blocks[:5]:  # Show first 5
            print(f"\n{Fore.RED}Missing text:{Style.RESET_ALL} {text[:100]}...")
            print(f"{Fore.YELLOW}Tag hierarchy:{Style.RESET_ALL}")
            
            for level, tag in enumerate(reversed(hierarchy)):
                indent = "  " * level
                classes = " ".join(tag['classes'])
                attrs = " ".join(f"{k}={v}" for k,v in tag['attrs'].items())
                print(f"{indent}<{tag['name']} class='{classes}' {attrs}>")
                
                if pattern_extractor:
                    # Try to diagnose why this didn't match any patterns
                    for category, pattern in pattern_extractor.category_patterns.items():
                        root_tag = BeautifulSoup(f"<{tag['name']}></{tag['name']}>", 'html.parser').find()
                        for cls in tag['classes']:
                            root_tag['class'] = root_tag.get('class', []) + [cls]
                        for k, v in tag['attrs'].items():
                            root_tag[k] = v
                            
                        matches = pattern_extractor._matches_pattern(root_tag, pattern)
                        if not matches:
                            matchers = next(iter(pattern.root_matchers))
                            print(f"{indent}  {Fore.BLUE}Failed {category} pattern:{Style.RESET_ALL}")
                            if tag['name'] != matchers.tag:
                                print(f"{indent}    - Tag mismatch: expected {matchers.tag}, got {tag['name']}")
                            if not set(tag['classes']).issuperset(matchers.classes):
                                print(f"{indent}    - Missing required classes: {matchers.classes - set(tag['classes'])}")
                            if matchers.required_children:
                                print(f"{indent}    - Missing required child elements")
            
        if len(missing_blocks) > 5:
            print(f"\n... and {len(missing_blocks)-5} more missing blocks")