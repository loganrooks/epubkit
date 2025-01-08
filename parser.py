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
        
        # Configure default cursor
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

        # Add mouse motion binding for hover effect
        self.text_display.tag_configure("hover", 
            background="light pink",
            font=("TkDefaultFont", 12, "bold"))
        self.text_display.bind("<Motion>", self.handle_hover)
        
        
    def load_epub(self):
        """Load and display EPUB content with preserved HTML structure"""
        book = epub.read_epub(self.epub_path)
        self.text_display.delete('1.0', tk.END)
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Process each text block
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = element.get_text().strip()
                if text:
                    # Store HTML structure
                    self.html_map[text] = {
                        'tag': element.name,
                        'classes': element.get('class', []),
                        'id': element.get('id', ''),
                        'html': str(element)
                    }
                    # Add text with newlines for visual separation
                    self.text_display.insert(tk.END, text + "\n\n")
    
    def handle_hover(self, event):
        """Handle mouse hover over text with improved coordinate handling"""
        index = self.text_display.index(f"@{event.x},{event.y}")
        
        # Get block boundaries considering both X and Y coordinates
        block_start = self.text_display.index(f"{index} linestart")
        block_end = self.text_display.index(f"{index} lineend")
        
        # Get current block bbox (x1,y1,x2,y2)
        try:
            bbox = self.text_display.bbox(index)
            if not bbox:
                return
            
            x1, y1, width, height = bbox
            x2 = x1 + width
            
            # Check if cursor is within text block boundaries
            if (x1 <= event.x <= x2):
                text = self.text_display.get(block_start, block_end).strip()
                
                if text in self.html_map:
                    # Remove previous highlight if different block
                    if self.current_hover_tag != text:
                        if self.current_hover_tag:
                            self.text_display.tag_remove("hover", "1.0", tk.END)
                        
                        # Add new highlight
                        self.text_display.tag_add("hover", block_start, block_end)
                        self.text_display.configure(cursor="hand2")
                        self.current_hover_tag = text
                return
                
            # Reset if cursor outside any text block
            if self.current_hover_tag:
                self.text_display.tag_remove("hover", "1.0", tk.END)
                self.text_display.configure(cursor="arrow")
                self.current_hover_tag = None
                
        except tk.TclError:
            pass

    def handle_leave(self, event):
        """Reset cursor when mouse leaves text widget"""
        if self.current_hover_tag:
            self.text_display.tag_remove("hover", "1.0", tk.END)
            self.text_display.configure(cursor="arrow")
            self.current_hover_tag = None
            
    def handle_click(self, event):
        """Handle text selection with confirmation"""
        try:
            # Get selection if there is one
            selected_text = self.text_display.get(tk.SEL_FIRST, tk.SEL_LAST).strip()
            if selected_text and selected_text in self.html_map:
                # Create confirmation dialog
                confirm = messagebox.askyesno(
                    "Confirm Selection",
                    f"Add this {self.current_selection_type} example?\n\n"
                    f"Text: {selected_text[:100]}...\n"
                    f"HTML Tag: {self.html_map[selected_text]['tag']}\n"
                    f"Classes: {', '.join(self.html_map[selected_text]['classes'])}"
                )
                
                if confirm:
                    # Convert HTML info to hashable format
                    html_info = (
                        self.html_map[selected_text]['tag'],
                        tuple(self.html_map[selected_text]['classes']),
                        self.html_map[selected_text]['id'],
                        self.html_map[selected_text]['html']
                    )
                    # Store text and HTML info as a hashable tuple
                    self.selected_tags[self.current_selection_type].add(
                        (selected_text, html_info)
                    )
                    self.status_label.config(
                        text=f"Added {self.html_map[selected_text]['tag']} "
                            f"to {self.current_selection_type}"
                    )
                    
        except tk.TclError:
            # No selection
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