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
                    # Store full tag hierarchy
                    tag_hierarchy = []
                    current = element
                    while current and current.name:
                        tag_info = {
                            'tag': current.name,
                            'classes': current.get('class', []),
                            'id': current.get('id', ''),
                            'attrs': {k:v for k,v in current.attrs.items() 
                                    if k not in ['class', 'id']}
                        }
                        tag_hierarchy.append(tag_info)
                        current = current.parent
                        
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
                    hierarchy_text = '\n'.join(
                        f"{'  '*i}{t['tag']}" +
                        (f".{'.'.join(t['classes'])}" if t['classes'] else "") +
                        (f"#{t['id']}" if t['id'] else "") +
                        (f"[{','.join(f'{k}={v}' for k,v in t['attrs'].items())}]" if t['attrs'] else "")
                        for i, t in enumerate(reversed(tag_hierarchy))
                    )
                    
                    confirm = messagebox.askyesno(
                        "Confirm Selection",
                        f"Add this {self.current_selection_type} example?\n\n"
                        f"Text: {text[:100]}...\n\n"
                        f"Tag Hierarchy:\n{hierarchy_text}"
                    )
                    
                    if confirm:
                        # Convert dictionary to hashable format using tag_hierarchy
                        html_info = tuple(
                            (t['tag'],
                            tuple(sorted(t['classes'])),
                            t['id'],
                            tuple((k, v) for k, v in sorted(t['attrs'].items())))
                            for t in tag_hierarchy
                        )
                        
                        self.selected_tags[self.current_selection_type].add((text, html_info))
                        self.status_label.config(
                            text=f"Added {tag_hierarchy[0]['tag']} to {self.current_selection_type}"
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
        self.log_path = Path("logs/pattern_generation")
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.setup_ui()
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
    def generate_patterns(self):
        """Generate regex patterns with negative lookaheads"""
        self.patterns = self.get_patterns_from_gpt()

    def format_examples_for_prompt(self) -> Dict[str, List[dict]]:
        examples = {}
        for category, items in self.selections.items():
            if not items:
                continue
                
            category_examples = []
            for text, html_info in items:
                # Get text snippets
                text_preview = {
                    'start': text[:5],
                    'end': text[-5:] if len(text) > 5 else text
                }
                
                # Convert tuple format back to full HTML structure
                html_structure = []
                current_html = ""
                for tag, classes, id, attrs in html_info:
                    tag_str = f"<{tag}"
                    if classes:
                        tag_str += f' class="{" ".join(classes)}"'
                    if id:
                        tag_str += f' id="{id}"'
                    for k, v in attrs:
                        tag_str += f' {k}="{v}"'
                    tag_str += ">"
                    current_html = tag_str + current_html + f"</{tag}>"
                    html_structure.append(current_html)
                
                category_examples.append({
                    'text_preview': text_preview,
                    'html': current_html
                })
            examples[category] = category_examples
        return examples

    def generate_gpt_prompt(self, examples: Dict[str, List[dict]]) -> str:
        prompt = ("Generate regex patterns that match HTML structures and capture the text content. "
                 "Patterns must match their category's structure exactly and not match other categories.\n\n")
        
        for category, examples_list in examples.items():
            prompt += f"\n{category.upper()} examples:\n"
            for ex in examples_list:
                prompt += (f"- HTML: {ex['html']}\n"
                         f"  Text starts: {ex['text_preview']['start']}, ends: {ex['text_preview']['end']}\n")
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

    def test_pattern(self, category, pattern_widget, results_widget):
        pattern = pattern_widget.get("1.0", tk.END).strip()
        results_widget.delete("1.0", tk.END)
        
        try:
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
            
            # Log results
            self.log_interaction(
                self.last_prompt,
                self.last_response,
                results
            )
            
            # Display results
            self._display_results(results_widget, results)
            
        except re.error as e:
            results_widget.insert("1.0", f"Invalid pattern: {str(e)}\n")

    def get_patterns_from_gpt(self) -> Dict[str, str]:
        """Get regex patterns using GPT-4 chat completion"""
        examples = self.format_examples_for_prompt()
        
        try:
            client = OpenAI()
            
            # Create messages for chat completion
            messages = [
                {
                    "role": "system", 
                    "content": "You are a regex pattern generation assistant. Generate precise patterns that match HTML structures."
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
                max_tokens=500
            )
            
            # Parse response from new format
            patterns = {}
            response_text = response.choices[0].message.content
            
            for line in response_text.strip().split('\n'):
                if ':' in line:
                    category, pattern = line.split(':', 1)
                    patterns[category.lower().strip()] = pattern.strip()
            
            return patterns
            
        except Exception as e:
            print(f"Error getting patterns from GPT: {e}")
            return {}

    def setup_ui(self):
        self.generate_patterns()
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        for category, pattern in self.patterns.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=category.title())
            
            # Pattern editor
            ttk.Label(frame, text="Pattern:").pack(pady=5)
            pattern_text = tk.Text(frame, height=4)
            pattern_text.insert("1.0", pattern)
            pattern_text.pack(fill=tk.X, padx=5)
            
            # Test results
            ttk.Label(frame, text="Matching Blocks:").pack(pady=5)
            results = tk.Text(frame, height=20)
            results.pack(fill=tk.BOTH, expand=True, padx=5)
            
            # Test button
            ttk.Button(
                frame,
                text="Test Pattern",
                command=lambda p=pattern_text, r=results: self.test_pattern(category, p, r)
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