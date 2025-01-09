from __future__ import annotations
import dataclasses
from functools import  singledispatchmethod
import logging
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
    Protocol, cast, runtime_checkable, Final, TYPE_CHECKING
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

class EPUBTagSelector:
    def __init__(self, epub_path: str):
        self.root = tk.Tk()
        self.root.title("EPUB Tag Selector")
        self.root.geometry("800x600")
        # self.cursor_path = os.path.join(
        #     os.path.dirname(os.path.abspath(__file__)), 
        #     "resources/cursors"
        # )      
        
        self.epub_path = epub_path
        self.selected_tags: CategoryDict[Tuple[str, Tuple[ImmutableTagInfo, ...]]] = {
            'headers': [],
            'subheaders': [],
            'toc': [],
            'footnotes': [],
            'body': []
        }
        
        self.current_selection_type: CategoryType = 'headers'
        self.html_map: Dict[str, HTMLInfo] = {}  # Maps text content to HTML structure
        self.current_hover_tag: Optional[str] = None
        self.setup_ui()
        self.load_epub()
        
    def setup_ui(self) -> None:
        # Left panel for content
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
       # Text widget for displaying content
        self.text_display = tk.Text(self.content_frame, wrap=tk.WORD)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        self.text_display.bind("<Button-1>", self.handle_click)
        self.text_display.bind("<Motion>", self.handle_hover)
        self.text_display.bind("<Leave>", self.handle_leave)  # Add leave handler
        # Add mouse motion binding for hover effect
        self.text_display.tag_configure("hover", 
            background="light pink",
            font=("TkDefaultFont", 12, "bold"))

       
        # Set default small cursor
        # cursor_spec = "@{}/arrow.xbm {}/arrow_mask.xbm black white".format(
        #     self.cursor_path, 
        #     self.cursor_path
        # )

        # self.text_display.configure(cursor=cursor_spec)
        self.text_display.configure(cursor="arrow")


        
        # Right panel for controls
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Selection type buttons
        ttk.Label(self.control_frame, text="Select examples of:").pack(pady=5)
        
        selection_types = [
            ("Main Headers", "headers"),
            ("Sub-headers", "subheaders"),
            ("Table of Contents", "toc"),
            ("Footnotes", "footnotes"),
            ("Body Text", "body")
        ]
        
        for text, value in selection_types:
            ttk.Button(
                self.control_frame, 
                text=text,
                command=lambda v=value: self.set_selection_type(v)
            ).pack(fill=tk.X, padx=5, pady=2)
            
        # Done button
        ttk.Button(
            self.control_frame,
            text="Done",
            command=self.finish_selection
        ).pack(fill=tk.X, padx=5, pady=20)
        
        # Status label
        self.status_label = ttk.Label(self.control_frame, text="Select main headers")
        self.status_label.pack(pady=5)


    def load_epub(self) -> None:
        """Load EPUB content with proper spacing"""
        book = epub.read_epub(self.epub_path)
        self.text_display.delete('1.0', tk.END)
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup: BeautifulSoup = BeautifulSoup(item.get_content(), 'html.parser')
            
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = element.get_text().strip()
                if text:
                    # Store complete tag hierarchy including spans
                    tag_hierarchy = []
                    current = element
                    
                    # First, collect all spans inside the main element
                    spans = element.find_all('span', recursive=True)
                    inner_tags = []
                    
                    # Start with innermost span and work outwards
                    for span in spans:
                        tag_info = TagInfo(
                            'span',
                            span.get('class', []),
                            span.get('id', ''),
                            tuple((k, v) for k, v in span.attrs.items() 
                                if k not in ['class', 'id'])
                        )
                        inner_tags.append(tag_info)
                    
                    # Add the main element (p, h1, etc.)
                    tag_info = TagInfo(
                        element.name,
                        tuple(element.get('class', [])),
                        element.get('id', ''),
                        tuple((k, v) for k, v in element.attrs.items() 
                            if k not in ['class', 'id'])
                    )
                    
                    # Combine main element with inner spans
                    tag_hierarchy = [tag_info] + inner_tags
                    
                    # Store in html_map
                    self.html_map[text] = {
                        'tag_hierarchy': tag_hierarchy,
                        'html': str(element)
                    }
                    
                    # Insert text with tag for entire block
                    block_start = self.text_display.index("end-1c")
                    self.text_display.insert(tk.END, text)
                    block_end = self.text_display.index("end-1c")
                    
                    # Add text block tag
                    self.text_display.tag_add(f"block_{len(self.html_map)}", 
                                            block_start, block_end)
                    self.text_display.insert(tk.END, "\n\n")

    def handle_hover(self, event: tk.Event) -> None:
        """Handle hover with block tags"""
        index = self.text_display.index(f"@{event.x},{event.y}")
        
        try:
            # Get tags at current position
            tags = self.text_display.tag_names(index)
            block_tags = [t for t in tags if t.startswith('block_')]
            
            if block_tags:
                # Get full block range instead of just line
                for tag in block_tags:
                    ranges = self.text_display.tag_ranges(tag)
                    if ranges:
                        block_start, block_end = ranges[0], ranges[1]
                        text = self.text_display.get(block_start, block_end).strip().strip("\n\n")
                        
                        if text and text in self.html_map:
                            if self.current_hover_tag != text:
                                self._reset_hover()
                                self.text_display.tag_add("hover", block_start, block_end)
                                self.text_display.configure(cursor="hand2")
                                self.current_hover_tag = text
                            return
            
            self._reset_hover()
                
        except tk.TclError:
            self._reset_hover()

    def _reset_hover(self) -> None:
        """Helper to reset hover state"""
        if self.current_hover_tag:
            self.text_display.tag_remove("hover", "1.0", tk.END)
            # cursor_spec = "@{}/arrow.xbm {}/arrow_mask.xbm black white".format(
            #     self.cursor_path,
            #     self.cursor_path
            # )
            # self.text_display.configure(cursor=cursor_spec)
            self.text_display.configure(cursor="arrow")

            self.current_hover_tag = None

    def handle_leave(self, event: tk.Event) -> None:
        """Reset cursor when mouse leaves text widget"""
        if self.current_hover_tag:
            self.text_display.tag_remove("hover", "1.0", tk.END)
            # cursor_spec = "@{}/arrow.xbm {}/arrow_mask.xbm black white".format(
            #     self.cursor_path,
            #     self.cursor_path
            # )
            # self.text_display.configure(cursor=cursor_spec)
            self.text_display.configure(cursor="arrow")
            self.current_hover_tag = None
            
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

    def handle_click(self, event: tk.Event) -> None:
        """Handle text selection with expanded hit area"""
        try:
            index = self.text_display.index(f"@{event.x},{event.y}")
            bbox = self.text_display.bbox(index)
            
            if not bbox:
                return
                
            x1, y1, width, height = bbox
            x2 = x1 + width
            y2 = y1 + height
            
            click_padding = 1
            
            if (x1 - click_padding <= event.x <= x2 + click_padding and 
                y1 - click_padding <= event.y <= y2 + click_padding):
                
                line_start = self.text_display.index(f"{index} linestart")
                line_end = self.text_display.index(f"{index} lineend")
                text = self.text_display.get(line_start, line_end).strip()
                
                if text and text in self.html_map:
                    # Format hierarchy text for display
                    tag_hierarchy = self.html_map[text]['tag_hierarchy']
                    hierarchy_text = self._format_hierarchy(tag_hierarchy, text)
                    
                    confirm = messagebox.askyesno(
                        "Confirm Selection",
                        f"Add this {self.current_selection_type} example?\n\n"
                        f"{hierarchy_text}"
                    )
                    
                    if confirm:
                        # Convert dictionary to hashable format using tag_hierarchy
                        html_info = tuple(ImmutableTagInfo.from_tag_info(tag) for tag in tag_hierarchy)
                        
                        self.selected_tags[self.current_selection_type].append((text, html_info))
                        self.status_label.config(
                            text=f"Added {tag_hierarchy[0][1]} to {self.current_selection_type}"
                        )
                        
        except tk.TclError:
            pass
            
    def set_selection_type(self, selection_type: CategoryType) -> None:
        """Change current selection type"""
        self.current_selection_type = selection_type
        self.status_label.config(
            text=f"Now selecting: {selection_type}"
        )
        
    def finish_selection(self) -> None:
        if not self.selected_tags['body']:
            self.status_label.config(text="Error: Must select at least one body text example")
            return
            
        # Show review dialog
        review = SelectionReviewDialog(self.root, self.selected_tags)
        self.root.wait_window(review.dialog)
        self.root.quit()
        
    def get_selections(self) -> CategoryDict[Tuple[str, List[TagInfo]]]:
        """Return the collected selections"""
        return self.selected_tags

class SelectionReviewDialog:
    selected_tags: CategoryDict[Tuple[str, List[TagInfo]]]
    html_map: Dict[str, HTMLInfo]
    root: tk.Tk
    status_label: ttk.Label

    def __init__(self, parent: tk.Tk, selections: CategoryDict[Tuple[str, List[TagInfo]]]) -> None:
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Review Selections")
        self.dialog.geometry("800x600")
        self.selections = selections
        self.listboxes: Dict[str, tk.Listbox] = {}
        self.count_labels: Dict[CategoryType, ttk.Label] = {}  # Track count labels
        self.setup_ui()
        self.setup_keyboard_shortcuts()
    
    def setup_keyboard_shortcuts(self) -> None:
        self.dialog.bind('<Delete>', self.handle_delete)
        
    def setup_ui(self) -> None:
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        for category, items in self.selections.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=category.title())
            
            # Control frame for buttons
            control_frame = ttk.Frame(frame)
            control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
            
            scrollbar = ttk.Scrollbar(frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, 
                               width=80, selectmode=tk.EXTENDED)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.listboxes[category] = listbox
            
            scrollbar.config(command=listbox.yview)
            
            for text, html_info in items:
                hierarchy_text = self._format_hierarchy(html_info)
                listbox.insert(tk.END, f"{text[:50]}...\n[{hierarchy_text}]\n")
            
            # Add buttons
            ttk.Button(
                control_frame,
                text="Delete Selected (Del)",
                command=lambda l=listbox, c=category: self.delete_selection(l, c)
            ).pack(side=tk.LEFT, padx=5)
            
            count_label = ttk.Label(
                control_frame,
                text=f"Items: {listbox.size()}"
            )

            count_label.pack(side=tk.RIGHT, padx=5)
            self.count_labels[category] = count_label
        
        ttk.Button(
            self.dialog,
            text="Confirm Selections",
            command=self.dialog.destroy
        ).pack(side=tk.BOTTOM, pady=10)

    def handle_delete(self, event: tk.Event) -> None:
        notebook = cast(ttk.Notebook, self.dialog.winfo_children()[0])
        current_tab = cast(str, notebook.select())
        category = cast(str, notebook.tab(current_tab)['text'].lower())
        if category in self.listboxes:
            self.delete_selection(self.listboxes[category], category)

    def delete_selection(self, listbox: tk.Listbox, category: CategoryType) -> None:
        """Delete selected items with confirmation"""
        selection = listbox.curselection()
        if not selection:
            return
            
        count = len(selection)
        if messagebox.askyesno(
            "Confirm Deletion",
            f"Delete {count} selected item{'s' if count > 1 else ''}?"
        ):
            # Delete items
            for idx in sorted(selection, reverse=True):
                self.selections[category].pop(idx)
                listbox.delete(idx)
            
            # Update count using stored reference
            self.count_labels[category].config(
                text=f"Items: {listbox.size()}"
            )

    def _format_hierarchy(self, hierarchy_tuple):
        """Format hierarchy from tuple format"""
        return ' > '.join(
        f"{tag_info.tag}" + 
        (f".{'.'.join(tag_info.classes)}" if tag_info.classes else "") +
        (f"#{tag_info.id}" if tag_info.id else "") +
        (f"[{','.join(f'{k}={v}' for k,v in tag_info.attrs)}]" if tag_info.attrs else "")
        for tag_info in reversed(hierarchy_tuple)
    )
    
    def finish_selection(self) -> None:
        if not self.selected_tags['body']:
            self.status_label.config(text="Error: Must select at least one body text example")
            return
            
        review = SelectionReviewDialog(self.root, self.selected_tags)
        self.root.wait_window(review.dialog)
        
        # Show pattern review
        pattern_review = PatternReviewDialog(self.root, self.selected_tags, self.html_map)
        self.root.wait_window(pattern_review.dialog)
        
        self.root.quit()
        
class PatternReviewDialog:
    def __init__(self, parent: tk.Tk, 
                 selections: CategoryDict[Tuple[str, List[TagInfo]]], 
                 html_map: Dict[str, HTMLInfo]) -> None:
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Review Tag Patterns")
        self.dialog.geometry("1000x800")
        self.selections = selections
        self.html_map = html_map
        self.patterns = {}
        self.html_formatter = HTMLFormatter()
        self.log_path = Path("logs/pattern_generation")
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Generate and test patterns before UI
        self.last_prompt = ""
        self.last_response = ""
        self.test_results = {}
        self.extractor = HTMLCategoryExtractor(self.selections)
        self.run_initial_tests()
        self.setup_ui()

    def run_initial_tests(self) -> None:
        """Test pattern matching for each text block"""
        self.test_results = {}
        
        # First pass - find all matching patterns for each text block
        matches_by_text: Dict[str, Set[CategoryType]] = {}
        
        # Test each HTML block against all patterns
        for text, html_info in tuple(self.html_map.items())[:100]:
            html = html_info['html']
            extracted = self.extractor.extract_category(html)
            
            # Track which categories matched this text
            matching_categories = set()
            for category, matches in extracted.items():
                if any(text == match['text'] for match in matches):
                    matching_categories.add(category)
            
            if matching_categories:
                matches_by_text[text] = matching_categories
        
        # Second pass - organize results by category
        for category in self.selections:
            results = {
                'matches': [],  # Unique matches (only matched this category)
                'conflicts': []  # Matched multiple categories
            }
            
            for text, matched_categories in matches_by_text.items():
                if category in matched_categories:
                    if len(matched_categories) == 1:
                        results['matches'].append(text)
                    else:
                        results['conflicts'].append(
                            (text, list(matched_categories - {category}))
                        )
                        
            self.test_results[category] = results
        
        self.log_interaction(
            "BeautifulSoup Category Extraction",
            json.dumps({cat: pattern.to_dict() 
                    for cat, pattern in self.extractor.category_patterns.items()}, 
                    indent=2),
            self.test_results
        )

    def _display_results(self, results_widget: tk.Text, results: Dict[str, List]) -> None:
        """Display pattern matching results with conflict highlighting"""
        results_widget.delete("1.0", tk.END)
        
        # Configure tags for coloring
        results_widget.tag_configure("success", foreground="green")
        results_widget.tag_configure("error", foreground="red")
        
        # Show unique matches
        match_count = len(results['matches'])
        results_widget.insert(tk.END, f"=== Unique Matches ({match_count}) ===\n")
        for text in results['matches'][:30]:  # Show first 30
            results_widget.insert(tk.END, f"✓ {text[:100]}...\n\n", "success")
        
        # Show conflicting matches
        if results['conflicts']:
            results_widget.insert(tk.END, 
                f"\n=== Pattern Conflicts ({len(results['conflicts'])}) ===\n")
            for text, other_cats in results['conflicts'][:30]:  # Show first 30
                results_widget.insert(tk.END, 
                    f"❌ Also matches: {', '.join(other_cats)}\n{text[:100]}...\n\n", 
                    "error")
            
    def setup_ui(self) -> None:
        """Setup UI with pre-generated patterns and results"""
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        for category, pattern in self.patterns.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=category.title())
            
            # Pattern editor with results
            ttk.Label(frame, text="Pattern:").pack(pady=5)
            pattern_text = tk.Text(frame, height=4)
            pattern_text.insert("1.0", pattern)
            pattern_text.pack(fill=tk.X, padx=5)
            
            # Pre-populated results
            results_text = tk.Text(frame, height=20)
            results_text.pack(fill=tk.BOTH, expand=True, padx=5)
            
            if category in self.test_results:
                self._display_results(results_text, self.test_results[category])
            
            # Test button for re-testing
            ttk.Button(
                frame,
                text="Test Pattern",
                command=lambda p=pattern_text, r=results_text, c=category: 
                    self.test_pattern(c, p, r)
            ).pack(pady=5)


    def generate_patterns(self):
        """Generate regex patterns with negative lookaheads"""
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

    
    def test_pattern(self, category, pattern_widget, results_widget):
        """Test pattern against HTML content with category validation"""
        pattern = pattern_widget.get("1.0", tk.END).strip()
        results_widget.delete("1.0", tk.END)
        
        try:
            regex = re.compile(pattern, re.DOTALL)
            matches = []
            false_positives = []
            
            # Test against all HTML content
            for text, html_info in self.html_map.items():
                html = html_info['html']
                if regex.search(html):
                    # Check if this text block was selected in any category
                    found_in = None
                    for cat, selections in self.selections.items():
                        if any(sel[0] == text for sel in selections):
                            found_in = cat
                            break
                    
                    if found_in == category:
                        matches.append(text)
                    elif found_in:
                        false_positives.append((text, found_in))
                    else:
                        matches.append(text)
            
            # Display results
            results_widget.insert(tk.END, f"=== Matches ({len(matches)}) ===\n")
            for text in matches:
                results_widget.insert(tk.END, f"✓ {text[:100]}...\n\n")
            
            if false_positives:
                results_widget.insert(tk.END, f"\n=== False Positives ({len(false_positives)} ===\n")
                results_widget.tag_configure("error", foreground="red")
                for text, cat in false_positives:
                    results_widget.insert(tk.END, f"❌ Matches {cat}: {text[:100]}...\n\n", "error")
                
        except re.error as e:
            results_widget.insert("1.0", f"Invalid pattern: {str(e)}\n")

# @dataclass(frozen=True)
# class CategoryPattern:
#     category: CategoryType
#     required_tags: List[Dict[str, Set[str]]] = field(default_factory=list)
#     excluded_tags: List[Dict[str, Set[str]]] = field(default_factory=list)
#     file_position_pattern: Optional[str] = None

#     def __post_init__(self) -> None:
#         if not self.required_tags and not self.file_position_pattern:
#             raise ValueError("Category pattern must have required tags or file position pattern")

#     def to_dict(self) -> dict:
#         return {
#             'category': self.category,
#             'required_tags': [
#                 {k: list(v) for k, v in tag.items()}
#                 for tag in self.required_tags
#             ],
#             'excluded_tags': [
#                 {k: list(v) for k, v in tag.items()}
#                 for tag in self.excluded_tags
#             ],
#             'file_position_pattern': self.file_position_pattern
#         }

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
class PatternAnalysis:
    """Analysis of common and unique patterns across examples"""
    shared_structure: List[TagRequirement]  # Must have these
    unique_features: List[TagRequirement]   # Distinguishing features
    variable_parts: List[TagVariation]      # Parts that can vary
    position_requirements: PositionRules     # Ordering constraints

@dataclass(frozen=True)
class TagRequirement:
    """Individual tag matching requirement"""
    tag: str
    classes: FrozenSet[str]
    id_pattern: Optional[str]
    attrs: Tuple[Tuple[str, str], ...]
    match_criteria: MatchCriteria = field(default_factory=lambda: MatchCriteria())
    required: bool = True
    position: Optional[int] = None 

@dataclass(frozen=True)
class TagVariation:
    """Allowed variations in tag structure"""
    tag_options: FrozenSet[str]
    class_options: FrozenSet[FrozenSet[str]]
    id_patterns: FrozenSet[str]
    attr_patterns: FrozenSet[Tuple[str, str]]
    min_depth: int = 0
    max_depth: Optional[int] = None

@dataclass(frozen=True)
class AnalysisResult:
    common_tags: List[TagRequirement]  # Tags found in all examples
    unique_tags: List[TagRequirement]  # Tags that distinguish from other categories 
    variable_parts: List[TagVariation]  # Parts that can vary between examples
    ordering: Dict[str, List[int]]     # Required tag ordering/nesting

# @dataclass(frozen=True)
# class PositionRules:
#     """Rules for tag positioning in hierarchy"""
#     required_order: List[str]  # Tags must appear in this order
#     max_depth: Optional[int] = None  # Maximum nesting depth
#     depth_requirements: Dict[str, int] = field(default_factory=dict)  # Tag must be at specific depth
#     relative_positions: Dict[str, Set[str]] = field(default_factory=dict)  # Tag A must be ancestor of Tag B

# Step 2: Enhance PositionRules
@dataclass(frozen=True)
class PositionRules:
    sequential_position: Optional[int] = None  # Position among siblings
    required_order: List[str] = field(default_factory=list)
    max_depth: Optional[int] = None
    depth_requirements: Dict[str, int] = field(default_factory=dict)

    def freeze(self):
        # Convert lists to tuples
        object.__setattr__(self, 'required_order', tuple(self.required_order))
        return self
    
    def __hash__(self):
        return hash((self.sequential_position, self.required_order, self.max_depth, 
                     tuple(self.depth_requirements.items())))


@dataclass(frozen=True)
class MatchCriteria:
    """How to match a tag in the hierarchy"""
    depth_type: Literal['anywhere', 'exact_depth', 'relative_depth'] = 'anywhere'
    depth: Optional[int] = None  # Used for exact_depth and relative_depth
    position: Optional[int] = None  # Sequential position in tag list
    must_be_direct_child: bool = False
    position_invariant: bool = False  # Added position_invariant flag

# Simplified TagMatcher focusing on core matching
@dataclass(frozen=True)
class TagMatcher:
    tag: str
    classes: FrozenSet[str] = field(default_factory=frozenset)
    id_pattern: Optional[str] = None
    attrs: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    match_criteria: MatchCriteria = field(default_factory=MatchCriteria)
    position_rules: Optional[PositionRules] = None

    def freeze(self):
        # Convert any sets to frozensets and lists to tuples
        object.__setattr__(self, 'classes', frozenset(self.classes))
        object.__setattr__(self, 'attrs', tuple(sorted(self.attrs)))
        self.position_rules.freeze()
        return self
    
    
    def unfreeze(self):
        # Convert any frozensets back to sets and tuples back to lists
        object.__setattr__(self, 'classes', set(self.classes))
        object.__setattr__(self, 'attrs', list(sorted(self.attrs)))
        return self


@dataclass(frozen=True)
class CategoryPattern:
    category: CategoryType
    root_matchers: Set[TagMatcher]
    global_excluded: Set[TagMatcher] = field(default_factory=set)
    file_position_pattern: Optional[str] = None
    variations: List[TagVariation] = field(default_factory=list)


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
        return self

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
        self._validate_selections()
        self.analyses: Dict[CategoryType, AnalysisResult] = {}

        self.category_patterns: Dict[CategoryType, CategoryPattern] = self._build_category_patterns()

    def _validate_selections(self) -> None:
        """Validate selections dictionary has required structure"""
        if not isinstance(self.selections, dict):
            raise ValueError("Selections must be a dictionary")
            
        # Check categories
        for category in self.selections:
            if category not in CATEGORIES:
                raise ValueError(f"Invalid category: {category}")
                
        # Check selection content structure
        for category, items in self.selections.items():
            if not isinstance(items, (list, tuple)):
                raise ValueError(f"Category {category} must contain a list/tuple")
                
            for item in items:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError(f"Invalid selection format in {category}")
                text, tag_info = item
                if not isinstance(text, str):
                    raise ValueError(f"Selection text must be string in {category}")
                if not isinstance(tag_info, tuple):
                    raise ValueError(f"Tag info must be tuple in {category}")
                
    def _validate_matcher(self, matcher: TagMatcher) -> None:
        """Validate matcher configuration"""
        if matcher.match_criteria.depth_type != 'anywhere':
            if matcher.match_criteria.depth is None:
                raise ValueError("Depth required for position-specific matching")
                
        if matcher.position is not None and matcher.position < 0:
            raise ValueError("Position must be non-negative")

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
        """Check if element matches category pattern"""
        # Check matchers (includes position/depth rules)
        
        for matcher in pattern.root_matchers:
            if not self._matches_tag(element, matcher):
                return False
                
            # Check position rules
            if matcher.position_rules:
                if not self._check_position_rules(element, matcher.position_rules):
                    return False

        # Check excluded patterns
        for matcher in pattern.global_excluded:
            if self._matches_tag(element, matcher):
                return False
                
        return True
        
    def _matches_tag_requirement(self, element: BeautifulSoup, req: TagRequirement) -> bool:
        """Check if element matches tag requirement"""
        # Handle position invariance
        match req.match_criteria.depth_type:
            case 'anywhere':
                # Search whole subtree
                return self._find_matching_tag_anywhere(element, req)
                
            case 'exact_depth':
                # Must be at specific depth from root
                return self._find_matching_tag_at_depth(element, req.match_criteria.depth)
                
            case 'relative_depth':
                # Must be N levels below current element
                return self._find_matching_tag_relative(element, req.match_criteria.depth)
        
    def _check_position_rules(self, element: bs4.Tag, rules: PositionRules) -> bool:
        # Check sequential position
        if rules.sequential_position is not None:
            siblings = list(element.parent.children if element.parent else [])
            if siblings.index(element) != rules.sequential_position:
                return False

        # Check depth requirements
        for tag, depth in rules.depth_requirements.items():
            if not self._verify_tag_depth(element, tag, depth):
                return False

        # Check tag ordering
        return self._verify_tag_order(element, rules.required_order)
    
    # def _find_matching_tag_anywhere(self, element: BeautifulSoup, req: TagRequirement) -> bool:
    #     """Find tag matching requirement anywhere in subtree"""
    #     if self._matches_basic_criteria(element, req):
    #         return True
            
    #     for child in element.find_all():
    #         if self._matches_basic_criteria(child, req):
    #             return True
    #     return False

    def _find_matching_tag_anywhere(self, element: BeautifulSoup, req: TagRequirement) -> bool:
        """Find tag matching requirement anywhere in tree with branch support"""
        MAX_DEPTH = 30
        visited = set()
        queue = [(element, 0)]  # (element, depth)
        
        while queue:
            current, depth = queue.pop(0)
            if depth > MAX_DEPTH:
                continue
                
            if id(current) in visited:
                continue
            visited.add(id(current))
                
            if self._matches_basic_criteria(current, req):
                return True
                
            # Add all children to queue regardless of nesting
            for child in current.find_all(recursive=False):
                queue.append((child, depth + 1))
                
        return False
    
    # def _find_matching_tag_anywhere(self, element: BeautifulSoup, req: TagRequirement) -> bool:
    #     """Find tag matching requirement anywhere in subtree with depth limit"""
    #     MAX_DEPTH = 100
        
    #     def search(el: BeautifulSoup, depth: int = 0) -> bool:
    #         if depth > MAX_DEPTH:
    #             return False
                
    #         if self._matches_basic_criteria(el, req):
    #             return True
                
    #         for child in el.children:
    #             if isinstance(child, bs4.Tag):
    #                 if search(child, depth + 1):
    #                     return True
    #         return False
            
    #     return search(element)
        
    def _find_matching_tag_at_depth(self, element: bs4.Tag, depth: int) -> bool:
        """Find tag at specific depth from root"""
        if depth is None:
            return True  # No depth requirement
            
        current_depth = 0
        current = element
        while current and current.parent:
            if current_depth == depth:
                return True
            current = current.parent
            current_depth += 1
        return False
        
    def _find_matching_tag_relative(self, element: BeautifulSoup, depth: int) -> bool:
        """Find tag at relative depth from current element"""
        current_depth = 0
        current = element
        for _ in range(depth):
            if not current.find_next():
                return False
            current = current.find_next()
            current_depth += 1
        return True
    
    def _verify_depth(self, element: BeautifulSoup, required_depth: int) -> bool:
        """Verify element is at required depth from root"""
        current_depth = 0
        current = element
        
        while current.parent:
            current = current.parent
            current_depth += 1
            
        return current_depth == required_depth

    def _get_depth(self, element: BeautifulSoup) -> int:
        """Get element depth handling broken nesting"""
        depth = 0
        visited = set()
        current = element
        
        while current and current.parent:
            if id(current) in visited:  # Detect cycles
                logging.warning(f"Circular parent reference detected at {current}")
                break
            visited.add(id(current))
            
            # Check if parent is still in scope
            if current.parent.find(current.name) != current:
                break
                
            depth += 1
            current = current.parent
            
        return depth
        
    def _matches_basic_criteria(self, element: BeautifulSoup, req: TagRequirement) -> bool:
        """Check basic tag criteria (tag, classes, id, attrs)"""
        if element.name != req.tag:
            return False
            
        element_classes = set(element.get('class', []))
        if req.classes and not req.classes.issubset(element_classes):
            return False
            
        element_id = element.get('id', '')
        if req.id_pattern and not re.match(req.id_pattern, element_id or ''):
            return False
            
        if req.attrs and not all(
            element.get(k) == v for k, v in req.attrs
        ):
            return False
            
        return True

    def _verify_tag_order(self, element: BeautifulSoup, required_order: List[str]) -> bool:
        """Verify tags appear in order handling broken nesting"""
        order_idx = 0
        visited = set()
        queue = [element]
        
        while queue and order_idx < len(required_order):
            current = queue.pop(0)
            
            if id(current) in visited:
                continue
            visited.add(id(current))
            
            if current.name == required_order[order_idx]:
                order_idx += 1
                
            # Add siblings and children
            queue.extend(current.find_next_siblings())
            queue.extend(current.find_all(recursive=False))
            
        return order_idx == len(required_order)
        
    def _verify_tag_depth(self, element: BeautifulSoup, tag: str, required_depth: int) -> bool:
        """Verify tag exists at required depth"""
        current_depth = 0
        current = element
        
        while current and current.parent and current != current.parent:
            if current_depth == required_depth and current.name == tag:
                return True
            current = current.parent
            current_depth += 1
        
        if current == current.parent:
            logging.warning(f"Element {element} is its own parent")

        return False
    
    def _build_css_selector(self, tag: str, classes: Set[str]) -> str:
        """Build CSS selector from tag and classes"""
        selector = tag
        if classes:
            class_selectors = [f".{cls}" for cls in classes]
            selector += "".join(class_selectors)
        return selector

    def _matches_tag(self, element: BeautifulSoup, matcher: TagMatcher) -> bool:
        """Match against BeautifulSoup Tag"""
        if not self._matches_basic_criteria(element, matcher):
            return False
            
        # Check match_criteria position and depth
        criteria = matcher.match_criteria
        
        # Check sequential position if specified
        if criteria.position is not None:
            siblings = list(element.parent.children)
            if siblings.index(element) != criteria.position:
                return False
                
        # Check depth requirements
        if criteria.depth is not None:
            if not self._verify_tag_depth(element, matcher.tag, criteria.depth):
                return False
                
        return True

    def _build_category_patterns(self) -> Dict[CategoryType, CategoryPattern]:
        patterns = {}
        for category, examples in self.selections.items():
            if not examples:
                continue

            
            common_tags=self._find_shared_structure(examples)
            unique_tags=self._find_distinguishing_features(category, shared=common_tags)
            variable_parts=self._identify_variations(examples, shared=common_tags)
            ordering=self._analyze_tag_ordering(examples)


            # Create analysis result
            analysis = AnalysisResult(
                common_tags=common_tags,
                unique_tags=unique_tags,
                variable_parts=variable_parts,
                ordering=ordering
            )
            self.analyses[category] = analysis

            # Build matchers from analysis
            root_matchers = set()
            for req in (analysis.common_tags + analysis.unique_tags):
                matcher = TagMatcher(
                    tag=req.tag,
                    classes=req.classes,
                    id_pattern=req.id_pattern,
                    attrs=req.attrs,
                    match_criteria=MatchCriteria(
                        depth_type='exact_depth',
                        depth=analysis.ordering.get(req.tag, [0])[0]
                    ),
                    position_rules=PositionRules(
                        sequential_position=req.position,
                        depth_requirements={
                            req.tag: analysis.ordering[req.tag][0]
                        }
                    )
                )
                root_matchers.add(matcher.freeze())
                
            patterns[category] = CategoryPattern(
                category=category,
                root_matchers=frozenset(root_matchers)
            )

        return patterns

    def _find_shared_structure(
    self,
    examples: List[Tuple[str, Tuple[ImmutableTagInfo, ...]]]
) -> List[TagRequirement]:
        """Find common tag patterns across all examples"""
        shared = []

        if not examples:
            return shared
            
        baseline = examples[0][1]
        
        for i, tag in enumerate(baseline):
            is_shared = True  # Initialize flag here
            
            req = TagRequirement(
                tag=tag.tag,
                classes=frozenset(tag.classes),
                id_pattern=tag.id if 'filepos' in tag.id else None,
                attrs=tag.attrs,
                position=i,  # Store sequential position
                match_criteria=MatchCriteria(
                    depth_type='exact_depth',
                    depth=i  # Store nesting depth in match criteria
                )
            )
            
            # Verify tag appears in same position in all examples
            for _, other_tags in examples[1:]:
                if i >= len(other_tags):
                    is_shared = False
                    break
                other = other_tags[i]
                if not (
                    other.tag == tag.tag and
                    set(other.classes) == set(tag.classes)
                ):
                    is_shared = False
                    break
                    
            if is_shared:
                shared.append(req)
                
        return shared

    def _find_distinguishing_features(
        self,
        category: CategoryType,
        shared: List[TagRequirement]
    ) -> List[TagRequirement]:
        """Find patterns unique to this category"""
        distinguishing = []
        
        # Compare against other categories
        for other_cat, other_items in self.selections.items():
            if other_cat == category or not other_items:
                continue
                
            # Find tags/classes unique to this category
            for req in shared:
                is_unique = True
                for _, other_tags in other_items:
                    for other in other_tags:
                        if (
                            other.tag == req.tag and
                            other.classes == req.classes
                        ):
                            is_unique = False
                            break
                    if not is_unique:
                        break
                        
                if is_unique:
                    distinguishing.append(req)
                    
        return distinguishing

    def _identify_variations(
        self,
        examples: List[Tuple[str, Tuple[ImmutableTagInfo, ...]]],
        shared: List[TagRequirement]
    ) -> List[TagVariation]:
        """Find allowed variations in tag structure"""
        variations = []
    
        # Track varying tag positions
        varying_positions = {}
        
        # Compare each example against shared structure
        for _, tags in examples:
            for i, tag in enumerate(tags):
                # Check if position is already covered by shared structure
                if any(s.position == i for s in shared):  # Changed from s.depth to s.position
                    continue
                    
                if i not in varying_positions:
                    varying_positions[i] = {
                        'tags': set(),
                        'classes': set(), 
                        'ids': set()
                    }
                    
                varying_positions[i]['tags'].add(tag.tag)
                varying_positions[i]['classes'].update(tag.classes)
                if tag.id:
                    varying_positions[i]['ids'].add(tag.id)
                    
        # Create variations for varying positions
        for depth, variants in varying_positions.items():
            variations.append(TagVariation(
                tag_options=frozenset(variants['tags']),
                class_options=frozenset(map(frozenset, variants['classes'])),
                id_patterns=frozenset(variants['ids']),
                attr_patterns=frozenset(),
                min_depth=depth,
                max_depth=depth
            ))
            
        return variations
    
    def _analyze_tag_ordering(
        self,
        examples: List[Tuple[str, Tuple[ImmutableTagInfo, ...]]]
    ) -> Dict[str, List[int]]:
        """Analyze where tags appear in hierarchy"""
        ordering = {}
        for _, tags in examples:
            for i, tag_info in enumerate(tags):
                if tag_info.tag not in ordering:
                    ordering[tag_info.tag] = []
                ordering[tag_info.tag].append(i)
        return ordering

    def _build_matcher_from_tag_info(
        self,
        tag_info: Tuple[ImmutableTagInfo, ...]
    ) -> Optional[TagMatcher]:
        """Convert tag info tuple to TagMatcher hierarchy"""
        if not tag_info:
            return None
            
        # Start with innermost tag and work outwards
        current_matcher = None
        for tag in reversed(tag_info):
            new_matcher = TagMatcher(
                tag=tag.tag,
                classes=frozenset(tag.classes),
                id_pattern=(fr"{tag.id}" if tag.id else None),
                attrs=tuple((k,v) for k,v in tag.attrs)
            )
            
            if current_matcher:
                # Add previous matcher as required child
                new_matcher = TagMatcher(
                    tag=new_matcher.tag,
                    classes=new_matcher.classes,
                    id_pattern=new_matcher.id_pattern,
                    attrs=new_matcher.attrs,
                    required_children=frozenset([current_matcher])
                )
            current_matcher = new_matcher
            
        return current_matcher



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

def get_user_tag_examples(epub_path: str) -> CategoryDict[Tuple[str, List[TagInfo]]]:
    """Run the GUI and return user selections"""
    selector = EPUBTagSelector(epub_path)
    selector.root.mainloop()
    if selector.selected_tags['body']:
        pattern_review = PatternReviewDialog(
            selector.root, 
            selector.selected_tags,
            selector.html_map
        )
        selector.root.wait_window(pattern_review.dialog)
    return selector.get_selections()
        
if __name__ == "__main__":
    epub_file = "resources/epubs/Being and Time - Martin Heidegger.epub"
    selections = get_user_tag_examples(epub_file)
    print("User selections:", selections)