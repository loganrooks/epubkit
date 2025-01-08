import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Set, Optional, Tuple
from openai import OpenAI
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



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
        self.selected_tags: Dict[str, Set[str]] = {
            'headers': set(),
            'subheaders': set(),
            'toc': set(),
            'footnotes': set(),
            'body': set()
        }
        
        self.current_selection_type = 'headers'
        self.html_map = {}  # Maps text content to HTML structure
        self.current_hover_tag = None
        self.setup_ui()
        self.load_epub()
        
    def setup_ui(self):
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


    def load_epub(self):
        """Load EPUB content with proper spacing"""
        book = epub.read_epub(self.epub_path)
        self.text_display.delete('1.0', tk.END)
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
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
                    for span in reversed(spans):
                        tag_info = (
                            'span',
                            span.get('class', []),
                            span.get('id', ''),
                            tuple((k, v) for k, v in span.attrs.items() 
                                if k not in ['class', 'id'])
                        )
                        inner_tags.append(tag_info)
                    
                    # Add the main element (p, h1, etc.)
                    tag_info = (
                        element.name,
                        element.get('class', []),
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
                    self.text_display.insert(tk.END, text + "\n\n")
                    block_end = self.text_display.index("end-1c")
                    
                    # Add text block tag
                    self.text_display.tag_add(f"block_{len(self.html_map)}", 
                                            block_start, block_end)

    def handle_hover(self, event):
        """Handle hover with block tags"""
        index = self.text_display.index(f"@{event.x},{event.y}")
        
        try:
            # Get tags at current position
            tags = self.text_display.tag_names(index)
            block_tags = [t for t in tags if t.startswith('block_')]
            
            if block_tags:
                block_start = self.text_display.index(f"{index} linestart")
                block_end = self.text_display.index(f"{index} lineend")
                text = self.text_display.get(block_start, block_end).strip()
                
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

    def _reset_hover(self):
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

    def handle_leave(self, event):
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

    def handle_click(self, event):
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
                        html_info = tuple(
                            (t[0], # tag
                            tuple(sorted(t[1])), # classes
                            t[2], # id
                            tuple((k, v) for k, v in sorted(t[3]))) # attrs
                            for t in tag_hierarchy
                        )
                        
                        self.selected_tags[self.current_selection_type].add((text, html_info))
                        self.status_label.config(
                            text=f"Added {tag_hierarchy[0][1]} to {self.current_selection_type}"
                        )
                        
        except tk.TclError:
            pass
            
    def set_selection_type(self, selection_type: str):
        """Change current selection type"""
        self.current_selection_type = selection_type
        self.status_label.config(
            text=f"Now selecting: {selection_type}"
        )
        
    def finish_selection(self):
        """Complete the selection process"""
        # Validate that we have required selections
        if not self.selected_tags['body']:
            self.status_label.config(
                text="Error: Must select at least one body text example"
            )
            return
            
        # Show review dialog
        review = SelectionReviewDialog(self.root, self.selected_tags)
        self.root.wait_window(review.dialog)
        self.root.quit()
        
    def get_selections(self) -> Dict[str, Set[str]]:
        """Return the collected selections"""
        return self.selected_tags

class SelectionReviewDialog:
    def __init__(self, parent, selections):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Review Selections")
        self.dialog.geometry("800x600")
        self.selections = selections
        self.listboxes = {}  # Store listbox references
        self.setup_ui()
        self.setup_keyboard_shortcuts()
    
    def setup_keyboard_shortcuts(self):
        self.dialog.bind('<Delete>', self.handle_delete)
        
    def setup_ui(self):
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
            
            ttk.Label(
                control_frame,
                text=f"Items: {listbox.size()}"
            ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            self.dialog,
            text="Confirm Selections",
            command=self.dialog.destroy
        ).pack(side=tk.BOTTOM, pady=10)

    def handle_delete(self, event):
        """Handle Delete key press"""
        notebook = self.dialog.winfo_children()[0]
        current_tab = notebook.select()
        category = notebook.tab(current_tab)['text'].lower()
        if category in self.listboxes:
            self.delete_selection(self.listboxes[category], category)

    def delete_selection(self, listbox, category):
        """Delete selected items with confirmation"""
        selection = listbox.curselection()
        if not selection:
            return
            
        count = len(selection)
        if messagebox.askyesno(
            "Confirm Deletion",
            f"Delete {count} selected item{'s' if count > 1 else ''}?"
        ):
            # Convert to list and reverse sort to delete from bottom up
            for idx in sorted(selection, reverse=True):
                item = list(self.selections[category])[idx]
                self.selections[category].remove(item)
                listbox.delete(idx)
            
            # Update item count
            listbox.master.winfo_children()[-1].config(
                text=f"Items: {listbox.size()}"
            )

    def _format_hierarchy(self, hierarchy_tuple):
        """Format hierarchy from tuple format"""
        return ' > '.join(
            f"{tag}" + 
            (f".{'.'.join(classes)}" if classes else "") +
            (f"#{id}" if id else "") +
            (f"[{','.join(f'{k}={v}' for k,v in attrs)}]" if attrs else "")
            for tag, classes, id, attrs in reversed(hierarchy_tuple)
        )
    
    def finish_selection(self):
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
    def __init__(self, parent, selections, html_map):
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

    def run_initial_tests(self):
        """Run initial tests to generate patterns and results"""
        
        for category in self.selections:
            results = {
                'matches': [],
                'false_positives': []
            }
            
            for text, html_info in self.html_map.items():
                html = html_info['html']
                extracted = self.extractor.extract_category(html)
                
                if category in extracted:
                    if any(text == match['text'] for match in extracted[category]):
                        results['matches'].append(text)
                    else:
                        # Check if this text belongs to another category
                        found_in = None
                        for cat, selections in self.selections.items():
                            if any(sel[0] == text for sel in selections):
                                found_in = cat
                                break
                        if found_in:
                            results['false_positives'].append((text, found_in))
            
            self.test_results[category] = results
        
        self.log_interaction(
            "BeautifulSoup Category Extraction",
            json.dumps(self.extractor.category_patterns, indent=2),
            self.test_results
        )

    def _display_results(self, results_widget, results):
        """Display pattern matching results in widget"""
        results_widget.delete("1.0", tk.END)
        
        # Configure tags for coloring
        results_widget.tag_configure("success", foreground="green")
        results_widget.tag_configure("error", foreground="red")
        
        if 'error' in results:
            results_widget.insert(tk.END, f"Error: {results['error']}\n", "error")
            return
            
        # Show matches
        results_widget.insert(tk.END, f"=== Matches ({len(results['matches'])}) ===\n")
        for text in results['matches']:
            results_widget.insert(tk.END, f"✓ {text[:100]}...\n\n", "success")
        
        # Show false positives
        if results['false_positives']:
            results_widget.insert(tk.END, 
                f"\n=== False Positives ({len(results['false_positives'])}) ===\n")
            for text, cat in results['false_positives']:
                results_widget.insert(tk.END, 
                    f"❌ Matches {cat}: {text[:100]}...\n\n", "error")

    def setup_ui(self):
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

    def log_interaction(self, prompt: str, response: str, results: dict):
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

    def test_pattern(self, category, pattern_widget, results_widget):
        """Test pattern with json string conversion"""
        json_pattern = pattern_widget.get("1.0", tk.END).strip()
        results_widget.delete("1.0", tk.END)
        
        try:
            pattern = self.json_to_regex(json_pattern)
            if not pattern:
                results_widget.insert("1.0", "Invalid pattern format")
                return
                
            results = {
                'matches': [],
                'false_positives': []
            }
            
            regex = re.compile(pattern, re.DOTALL)
            for text, html_info in self.html_map.items():
                html = html_info['html']
                if regex.search(html):
                    found_in = None
                    for cat, selections in self.selections.items():
                        if any(sel[0] == text for sel in selections):
                            found_in = cat
                            break
                    
                    if found_in == category:
                        results['matches'].append(text)
                    elif found_in:
                        results['false_positives'].append((text, found_in))
            
            self._display_results(results_widget, results)
            
        except Exception as e:
            results_widget.insert("1.0", f"Error: {str(e)}")

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

    def setup_ui(self):
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
                results_widget.insert(tk.END, f"\n=== False Positives ({len(false_positives)}) ===\n")
                results_widget.tag_configure("error", foreground="red")
                for text, cat in false_positives:
                    results_widget.insert(tk.END, f"❌ Matches {cat}: {text[:100]}...\n\n", "error")
                
        except re.error as e:
            results_widget.insert("1.0", f"Invalid pattern: {str(e)}\n")

class HTMLFormatter:
    @staticmethod
    def extract_immediate_tags(html_info) -> list:
        """Extract immediate containing tags from HTML structure"""
        immediate_tags = []
        current_nesting = []
        
        for tag, classes, tag_id, attrs in html_info:
            # Only process p and span tags
            if tag not in ['p', 'span']:
                continue
                
            # Build opening tag
            open_tag = f"<{tag}"
            if classes:
                open_tag += f' class="{" ".join(classes)}"'
            if tag_id:
                open_tag += f' id="{tag_id}"'
            for k, v in attrs:
                open_tag += f' {k}="{v}"'
            open_tag += ">"
            
            # Build closing tag
            close_tag = f"</{tag}>"
            
            immediate_tags.append((open_tag, close_tag))
        
        return list(reversed(immediate_tags))  # Reverse for proper nesting


@dataclass
class CategoryPattern:
    """Represents a category's unique tag pattern"""
    category: str
    required_tags: List[Dict[str, Set[str]]]  # List of {tag: required_classes}
    excluded_tags: List[Dict[str, Set[str]]]  # Tags that must not be present
    file_position_pattern: Optional[str] = None  # For matching filepos IDs if needed
    
    def to_dict(self) -> dict:
        """Convert to JSON serializable dictionary"""
        return {
            'category': self.category,
            'required_tags': [
                {k: list(v) for k, v in tag.items()}
                for tag in self.required_tags
            ],
            'excluded_tags': [
                {k: list(v) for k, v in tag.items()}
                for tag in self.excluded_tags
            ],
            'file_position_pattern': self.file_position_pattern
        }

class HTMLCategoryExtractor:
    def __init__(self, selections):
        self.selections = selections
        self.category_patterns: Dict[str, CategoryPattern] = self._build_category_patterns()
    
    def _build_category_patterns(self) -> Dict[str, CategoryPattern]:
        """Build unique identifiers for each category"""
        patterns = {}
        for category, items in self.selections.items():
            if not items:
                continue
            
            # Collect common required tags/classes
            required_tags = []
            excluded_tags = []
            file_position_pattern = None
            
            # Analyze tag signatures
            tag_signatures = self._collect_tag_signatures(items)
            
            # Find common required patterns
            for sig in tag_signatures:
                for tag, classes in sig.items():
                    if all(classes.issubset(other[tag]) 
                          for other in tag_signatures if tag in other):
                        required_tags.append({tag: classes})
                        
                    # Check for filepos pattern
                    if 'filepos' in str(classes):
                        file_position_pattern = r'filepos\d+'
            
            # Find excluded patterns (from other categories)
            for other_cat, other_items in self.selections.items():
                if other_cat != category:
                    other_sigs = self._collect_tag_signatures(other_items)
                    for sig in other_sigs:
                        for tag, classes in sig.items():
                            if not any(classes.issubset(req[tag]) 
                                     for req in required_tags if tag in req):
                                excluded_tags.append({tag: classes})
            
            patterns[category] = CategoryPattern(
                category=category,
                required_tags=required_tags,
                excluded_tags=excluded_tags,
                file_position_pattern=file_position_pattern
            )
            
        return patterns
    
    def _collect_tag_signatures(self, items):
        """Helper to collect tag signatures from items"""
        signatures = []
        for _, html_info in items:
            sig = {}
            for tag, classes, tag_id, _ in html_info:
                if tag in ['p', 'span']:
                    sig[tag] = set(classes)
            signatures.append(sig)
        return signatures
    
    def to_json(self) -> dict:
        """Convert patterns to JSON serializable format"""
        return {
            category: pattern.to_dict()
            for category, pattern in self.category_patterns.items()
        }

def get_user_tag_examples(epub_path: str) -> Dict[str, Set[str]]:
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