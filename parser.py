import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Set, Optional

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
            cursor_spec = "@{}/arrow.xbm {}/arrow_mask.xbm black white".format(
                self.cursor_path,
                self.cursor_path
            )
            self.text_display.configure(cursor=cursor_spec)
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
            
        self.root.quit()
        
    def get_selections(self) -> Dict[str, Set[str]]:
        """Return the collected selections"""
        return self.selected_tags
        
def get_user_tag_examples(epub_path: str) -> Dict[str, Set[str]]:
    """Run the GUI and return user selections"""
    selector = EPUBTagSelector(epub_path)
    selector.root.mainloop()
    return selector.get_selections()

if __name__ == "__main__":
    epub_file = "resources/Being and Time - Martin Heidegger.epub"
    selections = get_user_tag_examples(epub_file)
    print("User selections:", selections)