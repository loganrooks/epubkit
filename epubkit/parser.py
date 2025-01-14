from __future__ import annotations
from functools import  singledispatchmethod
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import bs4
import ebooklib  # type: ignore
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from typing import (
    Any, Dict, FrozenSet, List, Set, Optional, Tuple, TypedDict, NamedTuple, 
    Protocol, override, runtime_checkable, Final, TYPE_CHECKING
)
from openai import OpenAI  # type: ignore
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing_extensions import Literal
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if TYPE_CHECKING:
    from ebooklib import epub
    from openai import OpenAI

# Custom types and protocols
CategoryType = Literal["headers", "subheaders", "toc", "footnotes", "body"]

type HTML = str
type CategoryDict[T] = dict[CategoryType, List[T]]
type ImmutableCategoryDict[T] = dict[CategoryType, tuple[T]]

EMPTY_CATEGORY_DICT: Final[CategoryDict[Any]] = {
    "headers": [],
    "subheaders": [],
    "toc": [],
    "footnotes": [],
    "body": []
}


CATEGORIES: Final[List[CategoryType]] = ["headers", "subheaders", "toc", "footnotes", "body"]

class TagInfo(NamedTuple):
    tag: str
    classes: List[str]
    id: str
    attrs: List[Tuple[str, str]]  # Changed from tuple to List

@dataclass(frozen=True)
class ImmutableTagInfo:
    """Immutable version of TagInfo"""
    tag: str
    classes: Tuple[str, ...]
    id: str
    attrs: Tuple[Tuple[str, str], ...]
    
    @classmethod
    def from_tag_info(cls, tag_info: TagInfo) -> 'ImmutableTagInfo':
        return cls(
            tag=tag_info.tag,
            classes=tuple(tag_info.classes),
            id=tag_info.id,
            attrs=tuple(tag_info.attrs)
        )

class TestResult(TypedDict):
    matches: List[str]
    false_positives: List[tuple[str, str]]
    error: str

class HTMLInfo(TypedDict):
    tag_hierarchy: List[TagInfo]
    html: str

class TagSignature(TypedDict):
    tag: str
    classes: Set[str]
    id: str
    attrs: List[Tuple[str, str]]

class TestResult(TypedDict):
    matches: List[str]
    false_positives: List[Tuple[str, str]]

class CategoryMatch(TypedDict):
    text: str
    html: str

@runtime_checkable
class HTMLProcessor(Protocol):
    def extract_immediate_tags(self, html_info: List[TagInfo]) -> List[Tuple[str, str]]: ...
    def format_hierarchy(self, html_info: List[TagInfo], text: str) -> str: ...


class EPUBSelectorBackend:
    """Core tag selection logic without UI dependencies"""
    def __init__(self, epub_path: str | None = None):
        self.selected_tags: CategoryDict[Tuple[str, Tuple[ImmutableTagInfo, ...]]] = {
            'headers': [], 'subheaders': [], 'toc': [],
            'footnotes': [], 'body': []
        }
        
        self.html_map: Dict[str, HTMLInfo] = self.load_epub(epub_path) if epub_path else {}
        
    def load_epub(self, epub_path) -> Dict[str, HTMLInfo]:
        """Process EPUB and return text->HTML mapping"""
        book = epub.read_epub(epub_path)
        html_map = {}
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                text = element.get_text().strip()
                if text:
                    html_map[text] = self._process_element(element)
                    
        self.html_map = html_map
        return html_map
    
    def load_html(self, html: str) -> Dict[str, HTMLInfo]:
        """Process HTML string and return text->HTML mapping"""
        soup = BeautifulSoup(html, 'html.parser')
        html_map = {}
        
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = element.get_text().strip()
            if text:
                html_map[text] = self._process_element(element)
                
        self.html_map = html_map
        return html_map
    
    def _process_element(self, element: BeautifulSoup) -> HTMLInfo:
        """Process HTML element and return tag hierarchy"""
        inner_tags = [
            TagInfo('span', span.get('class', []), span.get('id', ''),
                   [(k,v) for k,v in span.attrs.items() if k not in ['class', 'id']])
            for span in element.find_all('span', recursive=True)
        ]
        
        main_tag = TagInfo(
            element.name,
            element.get('class', []),
            element.get('id', ''),
            [(k,v) for k,v in element.attrs.items() if k not in ['class', 'id']]
        )
        
        return {
            'tag_hierarchy': [main_tag] + inner_tags,
            'html': str(element)
        }

    def _extract_immediate_tags(self, hierarchy_tuple):
        """Extract immediate containing tags with no text between them"""
        immediate_tags = []
        for tag, classes, id, attrs in hierarchy_tuple:
            tag_str = f"<{tag}"
            if classes:
                tag_str += f' class="{" ".join(classes)}"'
            if id:
                tag_str += f' id="{id}"'
            for k, v in attrs:
                tag_str += f' {k}="{v}"'
            tag_str += ">"
            immediate_tags.append((tag_str, f"</{tag}>"))
        return immediate_tags

    def _format_hierarchy(self, hierarchy_tuple, text):
        """Format immediate nested tags for display"""
        tags = self._extract_immediate_tags(hierarchy_tuple)
        
        # Build nested structure
        html_preview = ""
        for open_tag, close_tag in tags:
            html_preview = f"{open_tag}{html_preview}{close_tag}"
        
        # Add text preview
        text_preview = f"Text starts: '{text[:5]}', ends: '{text[-5:]}'"
        
        return f"Structure: {html_preview}\n{text_preview}"
        
    def add_selection(self, text: str, category: CategoryType) -> bool:
        """Add selection to category"""
        if text not in self.html_map:
            return False
            
        tag_hierarchy = self.html_map[text]['tag_hierarchy']
        html_info = tuple(ImmutableTagInfo.from_tag_info(tag) for tag in tag_hierarchy)
        self.selected_tags[category].append((text, html_info))
        return True
        
    def get_selections(self) -> CategoryDict[Tuple[str, List[TagInfo]]]:
        """Return current selections"""
        return self.selected_tags

class SelectionReviewBackend:
    """Core selection review logic without direct Tk references."""
    selections: CategoryDict[List[Tuple[str, Tuple[ImmutableTagInfo, ...]]]]
    
    def __init__(self, selections):
        self.selections = selections

    def delete_items(self, category, indices):
        """Delete items at given indices from the selections."""
        for idx in sorted(indices, reverse=True):
            self.selections[category].pop(idx)

    def format_hierarchy(self, hierarchy_tuple):
        """Generate a hierarchy string from tag info tuples."""
        return ' > '.join(
            f"{t.tag}"
            + (f".{'.'.join(t.classes)}" if t.classes else "")
            + (f"#{t.id}" if t.id else "")
            + (f"[{','.join(f'{k}={v}' for k,v in t.attrs)}]" if t.attrs else "")
            for t in reversed(hierarchy_tuple)
        )

class PatternReviewBackend:
    """Handles pattern generation, testing, and logging without Tkinter references."""
    
    def __init__(self, selections: CategoryDict[Tuple[str, Tuple[ImmutableTagInfo, ...]]], html_map: Dict[str, HTMLInfo]):
        self.selections = selections
        self.html_map = html_map
        self.patterns = {}
        self.test_results = {}
        self.last_prompt = ""
        self.last_response = ""
        
        self.log_path = Path("logs/pattern_generation")
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Example placeholders – replace extractor/formatter with your actual backend usage
        self.extractor = HTMLCategoryExtractor(self.selections)

    def run_initial_tests(self) -> None:
        """Analyze each text block, figure out possible category matches."""
        self.test_results = {}
        matches_by_text: Dict[str, Set[str]] = {}

        # Compare extracted categories for each HTML block
        for text, html_info in tuple(self.html_map.items())[:100]:
            html = html_info['html']
            extracted = self.extractor.extract_category(html)

            matched_cats = set()
            for category, matches in extracted.items():
                if any(text == m['text'] for m in matches):
                    matched_cats.add(category)

            if matched_cats:
                matches_by_text[text] = matched_cats

        # Organize results
        for category in self.selections:
            results = {'matches': [], 'conflicts': [], 'false_positives': []}
            for text, cats in matches_by_text.items():
                if category in cats:
                    if len(cats) == 1:
                        results['matches'].append(text)
                    else:
                        # It's matched more than one category
                        results['conflicts'].append((text, list(cats - {category})))
            self.test_results[category] = results

        # Example logging
        self.log_interaction(
            "Initial Category Extraction",
            json.dumps({cat: pat.to_dict() for cat, pat in self.extractor.category_patterns.items()}, indent=2),
            self.test_results
        )

    def generate_patterns(self) -> None:
        """Generate or fetch patterns from GPT (or other service)."""
        self.patterns = self.get_patterns_from_gpt()

    def format_examples_for_prompt(self) -> str:
        formatted_examples = []
        
        for category, items in self.selections.items():
            if not items:
                continue
                
            formatted_examples.append(f"\n{category.upper()}:")
            category_examples = []
            
            for text, html_info in items:
                tags = HTMLFormatter.extract_immediate_tags(html_info)
                tag_structure = ""
                for open_tag, close_tag in tags:
                    tag_structure = f"{open_tag}{tag_structure}{close_tag}"
                preview = f"[Text start: {text[:5]}, end: {text[-5:]}]"
                category_examples.append(f"{tag_structure} {preview}")
            
            formatted_examples.append(",\n".join(category_examples))
        
        return "\n".join(formatted_examples)

    def generate_gpt_prompt(self, examples: Dict[str, List[dict]]) -> str:
        prompt = (
            "Generate regex patterns as a JSON string with the following requirements:\n"
            "1. Only match structure exactly for each category\n"
            "2. Do not match other categories' structures\n"
            "3. Capture text content between tags\n"
            "4. Response must be valid JSON with category names as keys\n"
            "5. No other text besides the JSON string\n"
            "6. Each field of the JSON string should match a given category.\n"
        
            "HTML structures to match:\n"
            f"{examples}\n"
        )
        return prompt

    def log_interaction(self, prompt: str, response: str, results: dict) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_path / f"pattern_gen_{timestamp}.json"
        
        log_data = {
            'timestamp': timestamp,
            'prompt': prompt,
            'response': response,
            'results': results
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def json_to_regex(self, json_pattern: str) -> str:
        """Convert JSON-encoded regex pattern to valid regex string"""
        try:
            # Remove outer quotes
            if json_pattern.startswith('"') and json_pattern.endswith('"'):
                pattern = json_pattern[1:-1]
            else:
                pattern = json_pattern
                
            # Convert JSON escapes to regex escapes
            pattern = pattern.replace('\\\\', '\\')  # Unescape backslashes
            pattern = pattern.replace('\\"', '"')    # Unescape quotes
            
            # Test pattern validity
            re.compile(pattern, re.DOTALL)
            return pattern
            
        except re.error as e:
            print(f"Invalid regex pattern: {e}")
            return None

    def clean_gpt_response(self, response_text: str) -> str:
        """Clean GPT response text before JSON parsing"""
        # Remove leading/trailing quotes if present
        if response_text.startswith("'") and response_text.endswith("'"):
            response_text = response_text[1:-1]
            
        # Remove markdown code block syntax
        response_text = response_text.replace('```json', '').replace('```', '')
        
        # Strip whitespace and newlines
        response_text = response_text.strip()
        
        return response_text

    def get_patterns_from_gpt(self) -> Dict[str, str]:
        """Get regex patterns using GPT-4 chat completion"""
        examples = self.format_examples_for_prompt()
        
        try:
            client = OpenAI()
            messages = [
                {
                    "role": "system",
                    "content": ("You are a regex pattern generator. "
                            "Respond only with a JSON string containing patterns. "
                            "No other text or explanation.")
                },
                {
                    "role": "user",
                    "content": self.generate_gpt_prompt(examples)
                }
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            self.last_response = response_text
            
            # Clean response before parsing
            cleaned_response = self.clean_gpt_response(response_text)
            self.patterns = json.loads(cleaned_response)
            
            return self.patterns
            
        except json.JSONDecodeError as e:
            print(f"Error parsing GPT response as JSON: {e}")
            self.log_interaction(
                self.last_prompt,
                response_text,  # Log original response
                {'error': str(e)}
            )
            return {}
        except Exception as e:
            print(f"Error getting patterns from GPT: {e}")
            return {}

    
    def test_pattern_logic(self, category: str, pattern: str):
        """
        Return (matches, false_positives) by testing 'pattern' across self.html_map.
        This is purely logic, no UI code.
        """
        try:
            regex = re.compile(pattern, re.DOTALL)
            matches, false_positives = [], []

            for text, html_info in self.html_map.items():
                if regex.search(html_info['html']):
                    found_in = None
                    for cat, items in self.selections.items():
                        if any(sel[0] == text for sel in items):
                            found_in = cat
                            break

                    if found_in == category:
                        matches.append(text)
                    elif found_in:
                        false_positives.append((text, found_in))
                    else:
                        matches.append(text)

            return matches, false_positives
        except re.error as e:
            return None, f"Regex error: {str(e)}"

    def log_interaction(self, prompt: str, response: str, results: dict) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_path / f"pattern_gen_{timestamp}.json"
        log_data = {
            'timestamp': timestamp,
            'prompt': prompt,
            'response': response,
            'results': results
        }
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

class CategoryExtractionError(Exception):
    """Raised when category extraction fails"""
    pass

class HTMLFormatter:
    @staticmethod
    def extract_immediate_tags(html_info: List[TagInfo]) -> List[Tuple[str, str]]:
        immediate_tags: List[Tuple[str, str]] = []
        
        for tag_info in html_info:
            if tag_info.tag not in ['p', 'span']:
                continue
                
            open_tag = f"<{tag_info.tag}"
            if tag_info.classes:
                open_tag += f' class="{" ".join(tag_info.classes)}"'
            if tag_info.id:
                open_tag += f' id="{tag_info.id}"'
            for k, v in tag_info.attrs:
                open_tag += f' {k}="{v}"'
            open_tag += ">"
            
            close_tag = f"</{tag_info.tag}>"
            immediate_tags.append((open_tag, close_tag))
        
        return list(reversed(immediate_tags))

@dataclass(frozen=True)
class ConditionalRule:
    """If condition_matcher matches, then required/excluded matchers apply"""
    condition_matcher: 'TagMatcher'
    required_matchers: FrozenSet['TagMatcher'] = field(default_factory=frozenset)
    excluded_matchers: FrozenSet['TagMatcher'] = field(default_factory=frozenset)

@dataclass(frozen=True)
class MatchCriteria:
    position_invariant: bool = False  # Child can be anywhere in hierarchy


@dataclass(frozen=True)
class TagMatcher:
    """Defines criteria for matching a tag in the hierarchy"""
    tag: str
    classes: FrozenSet[str] = field(default_factory=frozenset)
    id_pattern: Optional[str] = None
    attrs: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    required_children: FrozenSet['TagMatcher'] = field(default_factory=frozenset)
    excluded_children: FrozenSet['TagMatcher'] = field(default_factory=frozenset)
    conditional_rules: Tuple[ConditionalRule, ...] = field(default_factory=tuple)
    match_criteria: MatchCriteria = field(default_factory=MatchCriteria)

    def freeze(self):
        # Convert any sets to frozensets
        object.__setattr__(self, 'classes', frozenset(self.classes))
        object.__setattr__(self, 'required_children', frozenset(self.required_children))
        object.__setattr__(self, 'excluded_children', frozenset(self.excluded_children))
        object.__setattr__(self, 'conditional_rules', tuple(self.conditional_rules))
        object.__setattr__(self, 'attrs', tuple(sorted(self.attrs)))
        return self
    
    
    def unfreeze(self):
        # Convert any frozensets back to sets
        object.__setattr__(self, 'classes', set(self.classes))
        object.__setattr__(self, 'required_children', set(self.required_children))
        object.__setattr__(self, 'excluded_children', set(self.excluded_children))
        object.__setattr__(self, 'conditional_rules', tuple(self.conditional_rules))
        object.__setattr__(self, 'attrs', tuple(sorted(self.attrs)))
        return self
    
    def to_dict(self) -> dict:
        return {
            'tag': self.tag,
            'classes': list(self.classes),
            'id_pattern': self.id_pattern,
            'attrs': self.attrs,
            'required_children': [c.to_dict() for c in self.required_children],
            'excluded_children': [c.to_dict() for c in self.excluded_children],
            'conditional_rules': [
                {
                    'condition_matcher': rule.condition_matcher.to_dict(),
                    'required_matchers': [m.to_dict() for m in rule.required_matchers],
                    'excluded_matchers': [m.to_dict() for m in rule.excluded_matchers]
                }
                for rule in self.conditional_rules
            ],
            'match_criteria': self.match_criteria.__dict__
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)
        


@dataclass(frozen=True)
class CategoryPattern:
    category: CategoryType
    root_matchers: Set[TagMatcher]
    global_excluded: Set[TagMatcher] = field(default_factory=set)
    file_position_pattern: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.root_matchers and not self.file_position_pattern:
            raise ValueError("Category pattern must have root matchers or file position pattern")
        
    def freeze(self):
        # Convert any sets to frozensets
        object.__setattr__(self, 'root_matchers', frozenset(self.root_matchers))
        object.__setattr__(self, 'global_excluded', frozenset(self.global_excluded))
        return self
    
    def unfreeze(self):
        # Convert any frozensets back to sets
        object.__setattr__(self, 'root_matchers', set(self.root_matchers))
        object.__setattr__(self, 'global_excluded', set(self.global_excluded))
        return self

    def to_dict(self) -> dict:
        return {
            'category': self.category,
            'root_matchers': [m.to_dict() for m in self.root_matchers],
            'global_excluded': [m.to_dict() for m in self.global_excluded],
            'file_position_pattern': self.file_position_pattern
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


@dataclass(frozen=True)
class ParentSpec:
    """Specification for parent tag matching"""
    tag_options: FrozenSet[str] = frozenset()  # Empty means any tag
    class_options: FrozenSet[str] = frozenset() 
    id_patterns: FrozenSet[str] = frozenset()
    attr_patterns: FrozenSet[Tuple[str, str]] = frozenset()
    required: bool = False  # If True, must match one of the options

class HTMLCategoryExtractor:
    def __init__(self, selections: CategoryDict[Tuple[str, Tuple[ImmutableTagInfo, ...]]]) -> None:
        self.selections = selections
        self.category_patterns: Dict[CategoryType, CategoryPattern] = self._build_category_patterns()

    def extract_category(self, html_content: str) -> CategoryDict[CategoryMatch]:
        if len(html_content) == 0:
            matches = {cat: [] for cat in self.category_patterns}
            return matches
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Check if the parsed HTML is valid
            if not soup.find():
                raise CategoryExtractionError("Invalid HTML content")
            
            results: CategoryDict[CategoryMatch] = {}
            
            for category, pattern in self.category_patterns.items():
                matches: List[CategoryMatch] = []
                for root_matcher in pattern.root_matchers:
                    for root_element in soup.find_all(root_matcher.tag):
                        if self._matches_pattern(root_element, pattern):
                            matches.append({
                                'text': root_element.get_text().strip(),
                                'html': str(root_element)
                            })
                results[category] = matches
                
            return results
            
        except Exception as e:
            raise CategoryExtractionError(f"Failed to extract categories: {str(e)}")

    def _matches_pattern(self, element: BeautifulSoup, pattern: CategoryPattern) -> bool:
        """Check if element's complete hierarchy matches the pattern"""
        # Check file position pattern first
        if pattern.file_position_pattern:
            if not re.search(pattern.file_position_pattern, str(element)):
                return False
            
        tag_hierarchy = self._build_tag_hierarchy(element, pattern)
    
        
        # Check if any root matcher matches the hierarchy
        return any(
            self._matches_hierarchy(tag_hierarchy, root_matcher, pattern.global_excluded)
            for root_matcher in pattern.root_matchers
        )
    
    def _build_css_selector(self, tag: str, classes: Set[str]) -> str:
        """Build CSS selector from tag and classes"""
        selector = tag
        if classes:
            class_selectors = [f".{cls}" for cls in classes]
            selector += "".join(class_selectors)
        return selector

    def _build_tag_hierarchy(self, root_element: BeautifulSoup, root_matcher: TagMatcher | None = None) -> Dict[str, Any]:
        """Convert BeautifulSoup element to tag hierarchy dict, starting from root matcher if it exists in hierarchy"""
        
        result = {
            'tag': root_element.name,
            'classes': set(root_element.get('class', [])),
            'id': root_element.get('id', ''),
            'attrs': {k:v for k,v in root_element.attrs.items() 
                     if k not in ['class', 'id']},
            'children': []
        }
        
        for child in root_element.children:
            if isinstance(child, bs4.Tag):
                result['children'].append(self._build_tag_hierarchy(child))
                
        return result

    def _extract_enclosing_hierarchy(self, element: BeautifulSoup) -> List[dict]:
        """Extract only the tags that form the main enclosing hierarchy. For example in the html element seen here:
        <p class="calibre_6">
            <span class="calibre9">
                <span>1.<span class="italic"> '…als thematische Frage wirklicher Untersuchung'.</span></span>
            When Heidegger speaks of a question...</span></p>
        The enclosing hierarchy would be: p.calibre_6 > span.calibre9
        The next two spans are not part of the enclosing hierarchy.

        Another trickier example:
        <p id="filepos41760" class="calibre_10">
            <span class="calibre11">
                <span class="bold">
                    <span class="italic">
                        <a><span>1. The Necessity for Explicitly Restating the Question of Being</span></a><span>
        </span></span></span></span></p>
        The enclosing hierarchy would be: p.calibre_10 > span.calibre11 > span.bold > span.italic > a > span
        The <span></span> after the closing </a> tag is not part of the enclosing hierarchy.
        
        Args:
            element: BeautifulSoup element to analyze
            
        Returns:
            List of tag dictionaries representing the enclosing hierarchy from outer to inner
        """
        # Start with outermost tag
        current = element
        hierarchy = []
        
        while current:
            # Get tag info
            tag_info = {
                'tag': current.name,
                'classes': set(current.get('class', [])),
                'id': current.get('id', ''),
                'attrs': {k:v for k,v in current.attrs.items() 
                         if k not in ['class', 'id']}
            }
            
            # Check if this tag encloses the content
            # Get the last non-whitespace content
            last_content = None
            for content in current.stripped_strings:
                last_content = content
                
            if last_content:
                # Find where this content appears in the original HTML
                content_pos = str(current).find(last_content)
                if content_pos > -1:
                    # Check if there's a matching closing tag after this content
                    close_tag = f"</{current.name}>"
                    remaining_html = str(current)[content_pos + len(last_content):]
                    if close_tag in remaining_html:
                        # This tag is part of the enclosing hierarchy
                        hierarchy.append(tag_info)
            
            # Move to next tag
            current = next(current.children, None) if isinstance(current, bs4.Tag) else None
            
        return hierarchy

    def _matches_hierarchy(
        self, 
        hierarchy: Dict[str, Any], 
        matcher: TagMatcher,
        global_excluded: Set[TagMatcher]
    ) -> bool:
        """Recursively check if hierarchy matches the matcher pattern"""
        # Extract only enclosing tags
        enclosing_hierarchy = self._extract_enclosing_hierarchy(hierarchy)
        
        # Check global exclusions against enclosing hierarchy
        for excl in global_excluded:
            if any(self._matches_tag(tag, excl) for tag in enclosing_hierarchy):
                return False

        # Check if the enclosing hierarchy matches the pattern
        if not self._matches_tag(enclosing_hierarchy[0], matcher):
            return False
            
        # Check children only within enclosing hierarchy
        if matcher.required_children:
            for req in matcher.required_children:
                if not any(self._matches_tag(tag, req) for tag in enclosing_hierarchy[1:]):
                    return False
                    
        # Check excluded children
        if matcher.excluded_children:
            for excl in matcher.excluded_children:
                if any(self._matches_tag(tag, excl) for tag in enclosing_hierarchy[1:]):
                    return False
                    
        return True

    @singledispatchmethod
    def _matches_tag(self, hierarchy: Any, matcher: TagMatcher) -> bool:
        """Default implementation returns False for unknown types"""
        return False
    
    @_matches_tag.register
    def _(self, hierarchy: dict, matcher: TagMatcher) -> bool:
        """Match against dictionary representation of tag"""
        if hierarchy['tag'] != matcher.tag:
            return False
            
        if matcher.classes and not matcher.classes.issubset(hierarchy['classes']):
            return False
            
        if matcher.id_pattern and not re.match(matcher.id_pattern, hierarchy['id'] or ''):
            return False
            
        if matcher.attrs and not all(
            hierarchy['attrs'].get(k) == v for k, v in matcher.attrs
        ):
            return False
            
        return True
    
    @_matches_tag.register
    def _(self, hierarchy: bs4.Tag, matcher: TagMatcher) -> bool:
        """Match against BeautifulSoup Tag"""
        if hierarchy.name != matcher.tag:
            return False
            
        element_classes = set(hierarchy.get('class', []))
        if matcher.classes and not matcher.classes.issubset(element_classes):
            return False
            
        element_id = hierarchy.get('id', '')
        if matcher.id_pattern and not re.match(matcher.id_pattern, element_id or ''):
            return False
            
        if matcher.attrs and not all(
            hierarchy.get(k) == v for (k, v) in matcher.attrs
        ):
            return False
            
        return True

    def _build_category_patterns(self) -> Dict[CategoryType, CategoryPattern]:
        """Build category patterns from selections, considering only enclosing hierarchies"""
        patterns: Dict[CategoryType, CategoryPattern] = {}
        
        for category in self.selections:
            if len(self.selections[category]) == 0:
                continue

            root_matchers: Set[TagMatcher] = set()
            global_excluded: Set[TagMatcher] = set()

            # First build root matchers from enclosing hierarchies
            for text, html_info in self.selections[category]:
                # Convert the HTML string to BeautifulSoup to extract enclosing hierarchy
                soup = BeautifulSoup(f"<root>{text}</root>", 'html.parser')
                tag_hierarchy = self._extract_enclosing_hierarchy(soup)
                
                # Convert the enclosing hierarchy to ImmutableTagInfo format
                enclosing_tags = tuple(
                    ImmutableTagInfo(
                        tag=tag['tag'],
                        classes=tuple(sorted(tag['classes'])),
                        id=tag['id'],
                        attrs=tuple(sorted(tag['attrs'].items()))
                    ) for tag in tag_hierarchy
                )
                
                # Build matcher from enclosing tags only
                root_matcher = self._build_matcher_from_tag_info(enclosing_tags)
                if isinstance(root_matcher, TagMatcher):
                    root_matchers.add(root_matcher.freeze())

            # Build excluded patterns from distinct enclosing structures
            for other_cat, other_items in self.selections.items():
                if len(other_items) == 0 or other_cat == category:
                    continue
                    
                for text, other_info in other_items:
                    # Extract enclosing hierarchy for other category
                    soup = BeautifulSoup(f"<root>{text}</root>", 'html.parser')
                    other_hierarchy = self._extract_enclosing_hierarchy(soup)
                    
                    # Convert to ImmutableTagInfo
                    other_tags = tuple(
                        ImmutableTagInfo(
                            tag=tag['tag'],
                            classes=tuple(sorted(tag['classes'])),
                            id=tag['id'],
                            attrs=tuple(sorted(tag['attrs'].items()))
                        ) for tag in other_hierarchy
                    )
                    
                    other_matcher = self._build_matcher_from_tag_info(other_tags)
                    if isinstance(other_matcher, TagMatcher):
                        # Compare only enclosing hierarchies
                        is_distinct = True
                        for root_matcher in root_matchers:
                            if self._compare_enclosing_hierarchies(
                                other_matcher, root_matcher
                            ):
                                is_distinct = False
                                break
                        if is_distinct:
                            global_excluded.add(other_matcher.freeze())

            patterns[category] = CategoryPattern(
                category=category,
                root_matchers=root_matchers,
                global_excluded=global_excluded,
                file_position_pattern=(
                    r'filepos\d+' if any('filepos' in tag.id for tag in html_info)
                    else None
                )
            ).freeze()

        return patterns

    def _compare_enclosing_hierarchies(self, matcher1: TagMatcher, matcher2: TagMatcher) -> bool:
        """Compare two matchers considering only their enclosing structure"""
        if matcher1.tag != matcher2.tag or matcher1.classes != matcher2.classes:
            return False
            
        # Compare children recursively, but only those that are part of the enclosing hierarchy
        if len(matcher1.required_children) != len(matcher2.required_children):
            return False
            
        # Sort children to ensure consistent comparison
        m1_children = sorted(matcher1.required_children, key=lambda x: (x.tag, tuple(x.classes)))
        m2_children = sorted(matcher2.required_children, key=lambda x: (x.tag, tuple(x.classes)))
        
        return all(
            self._compare_enclosing_hierarchies(c1, c2)
            for c1, c2 in zip(m1_children, m2_children)
        )

    # ...existing code...

def html_to_selections(html_dict: CategoryDict[HTML]) -> CategoryDict[Tuple[str, Tuple[ImmutableTagInfo, ...]]]:
    """Convert dictionary of HTML strings to selections format"""
    selections: Dict[CategoryType, List[Tuple[str, Tuple[ImmutableTagInfo, ...]]]] = {
        cat: [] for cat in CATEGORIES
    }
    
    for category, html_list in html_dict.items():
        for html in html_list:
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text().strip()
            
            # Build tag hierarchy
            tag_infos = []
            current = soup.find()  # Get first element
            
            while current and hasattr(current, 'name'):
                if current.name in ['p', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    tag_info = ImmutableTagInfo(
                        tag=current.name,
                        classes=tuple(current.get('class', [])),
                        id=current.get('id', ''),
                        attrs=tuple(
                            (k, v) for k, v in current.attrs.items()
                            if k not in ['class', 'id']
                        )
                    )
                    tag_infos.append(tag_info)
                current = current.parent
            
            selections[category].append((text, tuple(reversed(tag_infos))))
            
    return selections


@dataclass
class TextBlock:
    text: str
    category: CategoryType
    header_path: List[str] = field(default_factory=list)  # Tracks nested header hierarchy
    footnotes: List[str] = field(default_factory=list)

@dataclass
class ExtractedText:
    headers: List[TextBlock]
    subheaders: List[TextBlock]
    body: List[TextBlock]
    footnotes: List[TextBlock]
    toc: List[TextBlock]
    
    def save_to_file(self, output_path: Path) -> None:
        """Save categorized text to files"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each category to separate file
        for category in CATEGORIES:
            blocks = getattr(self, category)
            if blocks:
                with open(output_path / f"{category}.txt", 'w', encoding='utf-8') as f:
                    for block in blocks:
                        f.write(f"{'#' * len(block.header_path)} {block.text}\n")
                        if block.footnotes:
                            f.write("\nFootnotes:\n")
                            for note in block.footnotes:
                                f.write(f"- {note}\n")
                        f.write("\n---\n\n")

def extract_categorized_text(epub_path: str, output_path: Optional[Path] = None) -> ExtractedText:
    """Extract and categorize text from EPUB file"""
    # Get user selections through GUI
    selections = get_user_tag_examples(epub_path)
    
    # Initialize extractor
    extractor = HTMLCategoryExtractor(selections)
    
    # Read EPUB file
    book = epub.read_epub(epub_path)
    extracted = ExtractedText([], [], [], [], [])
    current_headers: List[str] = []
    
    # Process each HTML document
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode('utf-8')
        categories = extractor.extract_category(content)
        
        # Process headers first to maintain hierarchy
        if categories['headers']:
            for match in categories['headers']:
                current_headers = [match['text']]  # Reset header stack for new main header
                extracted.headers.append(TextBlock(
                    text=match['text'],
                    category='headers',
                    header_path=current_headers.copy()
                ))
        
        # Process subheaders
        if categories['subheaders']:
            for match in categories['subheaders']:
                if current_headers:  # Only add if we have a parent header
                    header_path = current_headers + [match['text']]
                    extracted.subheaders.append(TextBlock(
                        text=match['text'],
                        category='subheaders',
                        header_path=header_path
                    ))
        
        # Process body text with current header context
        if categories['body']:
            for match in categories['body']:
                extracted.body.append(TextBlock(
                    text=match['text'],
                    category='body',
                    header_path=current_headers.copy()
                ))
        
        # Process footnotes
        if categories['footnotes']:
            for match in categories['footnotes']:
                extracted.footnotes.append(TextBlock(
                    text=match['text'],
                    category='footnotes'
                ))
    
    # Save to files if output path provided
    if output_path:
        extracted.save_to_file(output_path)
    
    return extracted

def extract_text_by_headers(
    epub_path: str,
    footnote_mode: Literal['ignore', 'inline', 'end'] = 'end',
    output_path: Optional[Path] = None
) -> Dict[str, str]:
    """
    Extract text organized by headers with configurable footnote handling
    
    Args:
        epub_path: Path to EPUB file
        footnote_mode: How to handle footnotes:
            - 'ignore': Skip footnotes
            - 'inline': Keep footnotes where they appear
            - 'end': Collect footnotes at end of each section
        output_path: Optional path to save organized text
    
    Returns:
        Dictionary mapping header paths to their content
    """
    extracted = extract_categorized_text(epub_path)
    organized_text: Dict[str, str] = {}
    
    # Group content by headers
    for header in extracted.headers:
        header_key = " > ".join(header.header_path)
        content_parts = [header.text]
        section_footnotes = []
        
        # Add subheaders and body text under this header
        for subheader in extracted.subheaders:
            if subheader.header_path[0] == header.text:
                content_parts.append(f"\n## {subheader.text}\n")
        
        for body in extracted.body:
            if body.header_path and body.header_path[0] == header.text:
                if footnote_mode == 'inline':
                    content_parts.append(body.text)
                    if body.footnotes:
                        content_parts.extend([f"\n[{i+1}] {note}" 
                                           for i, note in enumerate(body.footnotes)])
                else:
                    content_parts.append(body.text)
                    if body.footnotes:
                        section_footnotes.extend(body.footnotes)
        
        # Add footnotes according to mode
        if footnote_mode == 'end' and section_footnotes:
            content_parts.append("\nFootnotes:")
            content_parts.extend([f"\n[{i+1}] {note}" 
                                for i, note in enumerate(section_footnotes)])
        
        organized_text[header_key] = "\n".join(content_parts)
    
    # Save to file if path provided
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "organized_text.txt", 'w', encoding='utf-8') as f:
            for header, content in organized_text.items():
                f.write(f"# {header}\n\n{content}\n\n{'='*80}\n\n")
    
    return organized_text

if __name__ == "__main__":
    epub_file = "resources/epubs/Being and Time - Martin Heidegger.epub"
    output_dir = Path("output")
    
    # Example 1: Extract all categorized text
    print("Extracting categorized text...")
    categorized = extract_categorized_text(epub_file, output_dir / "categorized")
    
    # Example 2: Extract text organized by headers with footnotes at the end
    print("\nExtracting text by headers...")
    organized = extract_text_by_headers(
        epub_file,
        footnote_mode='end',
        output_path=output_dir / "by_headers"
    )
    
    print("\nExtraction complete. Check the output directory for results.")