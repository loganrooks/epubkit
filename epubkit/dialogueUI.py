from __future__ import annotations
from itertools import islice
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import font
from typing import Dict, List, Optional, TypedDict, get_args

from bs4 import BeautifulSoup
from tkhtmlview import HTMLLabel, HTMLScrolledText
from epubkit.debug import log_error, setup_logging
from epubkit.parser import EPUBSelectorBackend, ExtractedText, HTMLInfo, ImmutableTagInfo, PatternReviewBackend, CategoryType, SelectionReviewBackend, TOCEntry, TOCExtractor
import customtkinter as ctk
from html.parser import HTMLParser

setup_logging()

@dataclass
class DialogText:
    BUTTONS: dict = TypedDict("BUTTONS")
    TITLES: dict = TypedDict("TITLES")
    LABELS: dict = TypedDict("LABELS")
    MESSAGES: dict = TypedDict("MESSAGES")

    def __init__(self, **kwargs):
        for field in ['BUTTONS', 'TITLES', 'LABELS', 'MESSAGES']:
            if field in kwargs:
                setattr(self, field, kwargs[field])

    


class VaporwaveFormatter:
    """Converts text to vaporwave aesthetic"""
    
    FULLWIDTH_MAP = {chr(i): chr(i + 0xFEE0) for i in range(0x21, 0x7F)}
    KATAKANA_MAP = {
        'A': 'ア', 'E': 'エ', 'I': 'イ', 'O': 'オ', 'U': 'ウ',
        'KA': 'カ', 'KE': 'ケ', 'KI': 'キ', 'KO': 'コ', 'KU': 'ク',
        'SA': 'サ', 'SE': 'セ', 'SI': 'シ', 'SO': 'ソ', 'SU': 'ス',
        'TA': 'タ', 'TE': 'テ', 'TI': 'チ', 'TO': 'ト', 'TU': 'ツ',
        'NA': 'ナ', 'NE': 'ネ', 'NI': 'ニ', 'NO': 'ノ', 'NU': 'ヌ',
        'HA': 'ハ', 'HE': 'ヘ', 'HI': 'ヒ', 'HO': 'ホ', 'HU': 'フ',
        'MA': 'マ', 'ME': 'メ', 'MI': 'ミ', 'MO': 'モ', 'MU': 'ム',
        'YA': 'ヤ', 'YO': 'ヨ', 'YU': 'ユ',
        'RA': 'ラ', 'RE': 'レ', 'RI': 'リ', 'RO': 'ロ', 'RU': 'ル',
        'WA': 'ワ', 'WO': 'ヲ', 'N': 'ン'
    }

    @classmethod
    def to_vaporwave(cls, text: str, style: str = 'full') -> str:
        """Convert text to vaporwave aesthetic"""
        # Convert to uppercase for katakana mapping
        text = text.upper()
        
        # Full-width conversion
        fullwidth = ''.join(cls.FULLWIDTH_MAP.get(c, c) for c in text)
        
        if style == 'minimal':
            return fullwidth
            
        # Add spaces between characters
        spaced = ' '.join(fullwidth)
        
        # Create katakana version
        words = text.split()
        katakana = []
        for word in words:
            # Try to match syllables
            i = 0
            kana = []
            while i < len(word):
                if i < len(word) - 1 and word[i:i+2] in cls.KATAKANA_MAP:
                    kana.append(cls.KATAKANA_MAP[word[i:i+2]])
                    i += 2
                elif word[i] in cls.KATAKANA_MAP:
                    kana.append(cls.KATAKANA_MAP[word[i]])
                    i += 1
                else:
                    i += 1
            katakana.append(''.join(kana))
        
        katakana_text = ' '.join(katakana)
        
        return f"{katakana_text}  {spaced}"

    @classmethod
    def format_menu(cls, text: str) -> str:
        """Format text for menu items"""
        return cls.to_vaporwave(text, style='minimal')
        
    @classmethod
    def format_title(cls, text: str) -> str:
        """Format text for titles"""
        return cls.to_vaporwave(text, style='full')
    
    @classmethod
    def format_dialog_text(cls, dialog_class: type[DialogText]) -> type[DialogText]:
        """Create new class with vaporwave formatted text"""
        formatted = {}
        

        for field in ['TITLES', 'LABELS', 'BUTTONS', 'MESSAGES']:
            if hasattr(dialog_class, field):
                field_dict = getattr(dialog_class, field)
                formatted[field] = {
                    k: cls.format_title(v) if isinstance(v, str) else v
                    for k, v in field_dict.items()
                }


        # Create new class
        return type(
            f"Vaporwave{dialog_class.__name__}",
            (dialog_class,),
            formatted
        )


class DialogStyle:
    """Default style configuration"""
    BG_COLOR = "#ffffff"
    FG_COLOR = "#000000"
    ACCENT_COLOR = "#0078d7"
    FONT = ("TkDefaultFont", 11)
    BOLD_FONT = ("TkDefaultFont", 12, "bold")
    HOVER_STYLE = {
        "background": ACCENT_COLOR,
        "foreground": FG_COLOR,
        "font": BOLD_FONT
    }
    
    TEXT_STYLE = {
        "font": FONT,
        "bg": BG_COLOR,
        "fg": FG_COLOR
    }
    
    BUTTON_STYLE = {
        "font": FONT,
        "bg": BG_COLOR,
        "fg": FG_COLOR
    }



class BaseDialog(tk.Toplevel):
    def __init__(self, parent, title, style: DialogStyle = None):
        super().__init__(parent)
        self.style = style or DialogStyle()
        self.parent = parent
        self.title(title)

        # Track state
        self._is_setup = False
        self._cleanup_required = True
        
        # Window close protocol
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Apply base window style
        self.configure(bg=self.style.BG_COLOR)
        
        self.result = None
        
        self.wait_visibility()

        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Calculate minimum size after UI setup
        self.setup_ui()
        self._is_setup = True
        self.update_idletasks()

    def setup_ui(self):
        """Setup UI with style configuration"""
        pass

    def cleanup(self):
        """Clean up resources"""
        if self._cleanup_required:
            try:
                # Release modal state
                self.grab_release()
                
                # Clear any text widgets
                for widget in self.winfo_children():
                    if isinstance(widget, tk.Text):
                        widget.delete("1.0", tk.END)
                
                # Remove any temp files/resources
                
                self._cleanup_required = False
                
            except Exception as e:
                print(f"Error during cleanup: {e}")
                self._handle_error(e)
                raise

    def on_close(self):
        """Handle window close"""
        self.cleanup()
        self.destroy()

    def destroy(self):
        """Ensure cleanup on destroy"""
        self.cleanup()
        super().destroy()

    def _handle_error(e):
        """Handle exceptions in dialog"""
        log_error(e)


@dataclass
class EPUBTagSelectorText(DialogText):
    BUTTONS = TypedDict("BUTTON_TEXT", {
        "headers": str,
        "subheader": str,
        "toc": str,
        "footnotes": str,
        "body": str,
        "done": str
    }, total=True)
    
    TITLES = TypedDict("TEXT", {
        "main": str
    }, total=True)

    LABELS = TypedDict("LABELS", {
        "current_category": str
    })


class EPUBTagSelectorUI(BaseDialog):
    """UI layer for tag selection"""
    FIRST_N_BLOCKS = 300
    def __init__(self, parent, epub_path: str, title: str, style: DialogStyle = None, ui_text: EPUBTagSelectorText = None):
        self.backend = EPUBSelectorBackend(epub_path)
        self.ui_text = ui_text
        # UI state
        self.current_selection_type: CategoryType = 'headers'
        self.current_hover_tag: Optional[str] = None
        

        # Initialize UI
        super().__init__(parent, title, style=style)
        self.load_content(epub_path)

    


    def setup_ui(self):
        """Setup UI with style configuration"""
        # Left content panel
        self.content_frame = tk.Frame(self, bg=self.style.BG_COLOR)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Text display with style
        self.text_container = tk.Frame(self.content_frame)
        self.text_container.pack(fill=tk.BOTH, expand=True)
        # Add scrollbar first
        scrollbar = tk.Scrollbar(self.text_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        

        self.text_display = tk.Text(
            self.text_container,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=self.style.TEXT_STYLE["font"],
            bg=self.style.TEXT_STYLE["bg"],
            fg=self.style.TEXT_STYLE["fg"],
            padx=20,
            pady=10
        )
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Wire up events
        self.text_display.bind("<Button-1>", self.handle_click)
        self.text_display.bind("<Motion>", self.handle_hover)
        self.text_display.bind("<Leave>", self.handle_leave)
        
        # Configure hover effect
        self.text_display.tag_configure(
            "hover",
            background=self.style.ACCENT_COLOR,  # Cyan
            foreground=self.style.BG_COLOR  # Deep purple
        )

        self.text_display.configure(cursor="arrow")

        selection_types = [
            (self.ui_text.BUTTONS[category], category) for category in get_args(CategoryType)
        ]
        
        # Set minimum button width based on text
        max_text_width = max(len(text) for text, _ in selection_types)
        button_width = max(15, max_text_width + 4)  # Add padding
        

        # Right control panel
        self.control_frame = tk.Frame(self, width=button_width*2)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        
        # Selection type label
        tk.Label(
            self.control_frame, 
            text="セ ェ ク ツ S E L E C T I O N   T Y P E",
            font=self.style.TEXT_STYLE["font"],
            fg=self.style.SECONDARY_COLOR,
            bg=self.style.BG_COLOR
        ).pack(pady=5)
        
      
        # Selection type buttons
        for text, value in selection_types:
            tk.Button(
                self.control_frame, 
                text=text,
                width=button_width,  # Use calculated width
                command=lambda v=value: self.set_selection_type(v),
                **self.style.BUTTON_STYLE  # Use vaporwave button style
            ).pack(fill=tk.X, padx=5, pady=2)
            
        # Done button
        tk.Button(
            self.control_frame,
            text=self.ui_text.BUTTONS['done'],
            command=self.confirm,
            **self.style.BUTTON_STYLE  # Use vaporwave button style
        ).pack(fill=tk.X, padx=5, pady=20)
        
        # Status label
        self.status_label = tk.Label(
            self.control_frame, 
            text="S E L E C T セ ェ ク ツ\n{}".format(self.ui_text.LABELS[self.current_selection_type]),
            font=self.style.TEXT_STYLE["font"],
            wraplength=button_width*10,
            justify=tk.CENTER,
            fg=self.style.SECONDARY_COLOR,
            bg=self.style.BG_COLOR
        )
        self.status_label.pack(pady=5)

    def set_selection_type(self, selection_type: CategoryType) -> None: 
        """Set current selection type"""
        self.current_selection_type = selection_type
        self.status_label.config(text="S E L E C T セ ェ ク ツ\n{}".format(self.ui_text.LABELS[selection_type]))
        
    def load_content(self, epub_path):
        """Load and display content"""
        html_map = self.backend.load_epub(epub_path)
        self._populate_text_display(html_map)
        
    def _populate_text_display(self, html_map: Dict[str, dict]):
        """
        Populate the text display with each entry in html_map, tagging each
        block so the hover and click handlers can process them.
        """
        self.text_display.delete("1.0", tk.END)
        for i, (text_key, data) in enumerate(islice(html_map.items(), self.FIRST_N_BLOCKS)):
            block_start = self.text_display.index('end-1c')
            self.text_display.insert(tk.END, text_key)
            block_end = self.text_display.index('end-1c')
            block_tag = f"block_{i}"
            self.text_display.tag_add(block_tag, block_start, block_end)
            
            # Configure block tag to be below hover
            self.text_display.tag_lower(block_tag, "hover")
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
          
                        if text and text in self.backend.html_map:
                            if self.current_hover_tag != text:
                                self._reset_hover()
                                self.text_display.tag_add("hover", block_start, block_end)
                                self.text_display.configure(cursor="hand2")
                                self.current_hover_tag = text
                            return
            
            self._reset_hover()
                
        except tk.TclError as e:
            print(f"TclError: {e}")
            self._reset_hover()

    def _reset_hover(self) -> None:
        """Helper to reset hover state"""
        if self.current_hover_tag:
            self.text_display.tag_remove("hover", "1.0", tk.END)
            self.text_display.configure(cursor="arrow")

            self.current_hover_tag = None

    def handle_leave(self, event: tk.Event) -> None:
        """Reset cursor when mouse leaves text widget"""
        if self.current_hover_tag:
            self.text_display.tag_remove("hover", "1.0", tk.END)
            self.text_display.configure(cursor="arrow")
            self.current_hover_tag = None

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
                print(f"Clicked text: {text[:30]}...")
                
                if text and text in self.backend.html_map:
                    # Format hierarchy text for display
                    tag_hierarchy = self.backend.html_map[text]['tag_hierarchy']
                    hierarchy_text = self.backend._format_hierarchy(tag_hierarchy, text)
                    
                    confirm = messagebox.askyesno(
                        VaporwaveFormatter.format_title("Confirm Selection"),
                        f"Add this {self.current_selection_type} example?\n\n"
                        f"{hierarchy_text}"
                    )
                    
                    if confirm:
                        # Convert dictionary to hashable format using tag_hierarchy
                        html_info = tuple(ImmutableTagInfo.from_tag_info(tag) for tag in tag_hierarchy)
                        
                        self.backend.selected_tags[self.current_selection_type].append((text, html_info))
                        self.status_label.config(
                            text=VaporwaveFormatter.format_title(f"Added {tag_hierarchy[0][1]} to {self.ui_text.LABELS[self.current_selection_type]}")
                        )
                        
        except tk.TclError as e:
            print(f"TclError: {e}")

    def confirm(self):
        """Store result and close"""
        self.result = (self.backend.html_map, self.backend.selected_tags)
        self.cleanup()
        self.destroy()

@dataclass
class PatternReviewText(DialogText):
    """Text constants for pattern review dialog"""
    TITLES = {
        "main": "Pattern Review",
        "test": "Test Results",
        "error": "Error",
        "extraction": "Extraction"
    }
    
    LABELS = {
        "pattern": "Pattern:",
        "matches": "Matches",
        "conflicts": "Conflicts",
        "false_positives": "False Positives"
    }
    
    BUTTONS = {
        "test": "Test Pattern",
        "confirm": "Confirm Patterns"
    }
    
    MESSAGES = {
        "error": "Pattern Error",
        "no_matches": "No matches found"
    }

@dataclass
class SelectionReviewText(DialogText):
    """Text constants for selection review dialog"""
    TITLES = {
        "main": "Review Selections",
        "confirm": "Confirm Delete"
    }
    
    LABELS = {
        "total_items": "Total Items: {}",
        "items_count": "Items: {}/{}",
        "search": "Search..."
    }
    
    BUTTONS = {
        "delete": "Delete Selected (Del)",
        "confirm": "Confirm Selections"
    }
    
    MESSAGES = {
        "confirm_delete": "Delete {} selected item{}?"
    }

class PatternReviewDialog(BaseDialog):
    """Tkinter frontend that delegates pattern logic to PatternReviewBackend."""
    
    def __init__(self, parent: tk.Tk, title, selections, html_map, 
                 style: DialogStyle = None,
                 ui_text: PatternReviewText = None):
        self.ui_text = ui_text or PatternReviewText()
        self.backend = PatternReviewBackend(selections, html_map)

        self.backend.run_initial_tests()

        print("Patterns:", self.backend.patterns)  # Debug
        print("Test results:", self.backend.test_results)  # Debug

        
        super().__init__(parent, title=title, style=style)
        self.geometry("1000x800+1080+840")  # Add position
        
        # Generate patterns before UI setup
        # self.backend.generate_patterns()  # Add this line

    def setup_ui(self) -> None:
        # Configure ttk style for notebook
        style = ttk.Style()
        style.configure("Vapor.TNotebook", 
            background=self.style.BG_COLOR,
            foreground=self.style.FG_COLOR)
        style.configure("Vapor.TNotebook.Tab",
            background=self.style.BG_COLOR,
            foreground=self.style.SECONDARY_COLOR,
            padding=[10, 5],
            font=self.style.TEXT_STYLE["font"])

        # Use ttk.Notebook instead of tk.Frame
        self.notebook = ttk.Notebook(self, style="Vapor.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a tab for each category
        for category, pattern_str in self.backend.patterns.items():
            frame = ttk.Frame(self.notebook, style="Vapor.TNotebook")
            self.notebook.add(frame, text=category.title())
            
            input_frame = ttk.Frame(frame)
            input_frame.pack(fill=tk.X, padx=5, pady=5)

            # Pattern input
            tk.Label(input_frame, 
                    text=self.ui_text.LABELS["pattern"],
                    **self.style.TEXT_STYLE
            ).pack(side=tk.LEFT, pady=5)
            
            pattern_text = tk.Text(frame, height=4, width=50,
                                 **self.style.TEXT_STYLE)
            pattern_text.insert("1.0", pattern_str)
            pattern_text.pack(fill=tk.X, padx=5, pady=5)

            # Results display
            results_text = tk.Text(frame, height=20,
                                 **self.style.TEXT_STYLE)
            results_text.pack(fill=tk.BOTH, expand=True, padx=5)

            if category in self.backend.test_results:
                self._display_results(results_text, self.backend.test_results[category])

            # Test button
            tk.Button(frame,
                     text=self.ui_text.BUTTONS["test"],
                     command=lambda p=pattern_text, r=results_text, c=category:
                         self.test_pattern(c, p, r),
                     **self.style.BUTTON_STYLE
            ).pack(pady=5)

        # Add confirm button at bottom
        tk.Button(self,
                 text=self.ui_text.BUTTONS["confirm"],
                 command=self.confirm,
                 **self.style.BUTTON_STYLE
        ).pack(pady=10)

    def _display_results(self, results_widget: tk.Text, results: Dict[str, List]) -> None:
        """Show results from backend logic in a text widget."""
        results_widget.delete("1.0", tk.END)
        
        results_widget.tag_configure("success", foreground="green")
        results_widget.tag_configure("error", foreground="red")

        match_count = len(results['matches'])
        results_widget.insert(tk.END, f"=== {self.ui_text.LABELS['matches']} ({match_count}) ===\n")
        for text in results['matches'][:30]:
            results_widget.insert(tk.END, f"✓ {text[:100]}...\n\n", "success")
        
        if results['conflicts']:
            results_widget.insert(tk.END, f"\n=== {self.ui_text.LABELS['conflicts']} ({len(results['conflicts'])}) ===\n")
            for text, other_cats in results['conflicts'][:30]:
                results_widget.insert(
                    tk.END,
                    f"❌ {self.ui_text.LABELS['false_positives']}: {', '.join(other_cats)}\n{text[:100]}...\n\n",
                    "error"
                )

    def test_pattern(self, category: str, pattern_widget: tk.Text, results_widget: tk.Text):
        """Use backend logic to test pattern, then display results."""
        pattern = pattern_widget.get("1.0", tk.END).strip()
        results_widget.delete("1.0", tk.END)

        matches, error_or_falsepos = self.backend.test_pattern_logic(category, pattern)
        if matches is None:
            # It's an error (error_or_falsepos is a string)
            results_widget.insert("1.0", f"{self.ui_text.MESSAGES['error']}: {error_or_falsepos}\n")
            return

        false_positives = error_or_falsepos
        # Display matches
        results_widget.insert(tk.END, f"=== {self.ui_text.LABELS['matches']} ({len(matches)}) ===\n")
        for t in matches:
            results_widget.insert(tk.END, f"✓ {t[:100]}...\n\n")

        if false_positives:
            results_widget.insert(tk.END, f"\n=== {self.ui_text.LABELS['false_positives']} ({len(false_positives)}) ===\n")
            results_widget.tag_configure("error", foreground="red")
            for t, cat in false_positives:
                results_widget.insert(tk.END, f"❌ {self.ui_text.LABELS['false_positives']}: {cat}: {t[:100]}...\n\n", "error")

    def confirm(self):
        """Store patterns and close"""
        self.result = {
            category: tab.children['!text'].get("1.0", tk.END).strip()
            for category, tab in zip(self.backend.patterns.keys(), 
                                   self.notebook.tabs())
        }
        self.cleanup()
        self.destroy()

class SelectionReviewDialogUI(BaseDialog):
    """Dialog for reviewing and managing tag selections"""
    
    def __init__(self, parent: tk.Tk, title: str,
                 selections, 
                 style: DialogStyle = None,
                 ui_text: SelectionReviewText = None):
        self.ui_text = ui_text or SelectionReviewText()
        self.backend = SelectionReviewBackend(selections)
        
        self.listboxes: Dict[str, tk.Listbox] = {}
        self.count_labels: Dict[str, tk.Label] = {}
        self.search_vars: Dict[str, tk.StringVar] = {}
        
        super().__init__(parent, title=title, style=style)
        self.geometry("800x600+1080+840")
        self.setup_keyboard_shortcuts()

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.bind("<Control-a>", lambda e: self.select_all())
        self.bind("<Control-f>", lambda e: self.focus_search())
        self.bind("<Delete>", self.handle_delete)
        
    def setup_ui(self) -> None:
        """Setup tabbed interface with search and controls"""
        # Configure ttk style
        style = ttk.Style()
        style.configure("Vapor.TNotebook", 
            background=self.style.BG_COLOR,
            foreground=self.style.FG_COLOR)
        style.configure("Vapor.TNotebook.Tab",
            background=self.style.BG_COLOR,
            foreground=self.style.SECONDARY_COLOR,
            padding=[10, 5],
            font=self.style.TEXT_STYLE["font"])
        
        # Create notebook
        self.notebook = ttk.Notebook(self, style="Vapor.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        for category, items in self.backend.selections.items():
            frame = self._create_category_tab(self.notebook, category, items)
            self.notebook.add(frame, text=category.title())
            
        # Bottom controls
        controls = tk.Frame(self, bg=self.style.BG_COLOR)
        controls.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        tk.Button(
            controls, 
            text=self.ui_text.BUTTONS["confirm"],
            command=self.confirm,
            **self.style.BUTTON_STYLE
        ).pack(side=tk.RIGHT, padx=5)
        
        tk.Label(
            controls,
            text=self.ui_text.LABELS["total_items"].format(
                sum(len(items) for items in self.backend.selections.values())
            ),
            font=self.style.TEXT_STYLE["font"],
            fg=self.style.TEXT_STYLE["fg"],
            bg=self.style.TEXT_STYLE["bg"]
        ).pack(side=tk.LEFT, padx=5)

    def _create_category_tab(self, notebook: ttk.Notebook, category: str, items: List) -> ttk.Frame:
        """Create tab for category with search and list"""
        frame = ttk.Frame(notebook, style="Vapor.TNotebook")
        
        # Search bar
        search_frame = ttk.Frame(frame, style="Vapor.TNotebook")
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        search_var = tk.StringVar()
        self.search_vars[category] = search_var
        search_var.trace_add("write", lambda *_: self._filter_list(category))
        
        tk.Entry(
            search_frame,
            textvariable=search_var,
            width=40,
            font=self.style.TEXT_STYLE["font"],
            bg=self.style.TEXT_STYLE["bg"],
            fg=self.style.TEXT_STYLE["fg"]
        ).pack(side=tk.LEFT, padx=5)
        
        # List with scrollbar
        list_frame = ttk.Frame(frame, style="Vapor.TNotebook")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.EXTENDED,
            activestyle='none',
            font=self.style.TEXT_STYLE["font"],
            bg=self.style.TEXT_STYLE["bg"],
            fg=self.style.TEXT_STYLE["fg"]
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        self.listboxes[category] = listbox
        
        # Populate list
        self._populate_list(category, items)
        
        # Controls
        control_frame = ttk.Frame(frame, style="Vapor.TNotebook")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        tk.Button(
            control_frame,
            text=self.ui_text.BUTTONS["delete"],
            command=lambda: self.delete_selection(listbox, category),
            **self.style.BUTTON_STYLE
        ).pack(side=tk.LEFT)
        
        count_label = tk.Label(
            control_frame, 
            text=f"Items: {len(items)}",
            font=self.style.TEXT_STYLE["font"],
            fg=self.style.TEXT_STYLE["fg"],
            bg=self.style.TEXT_STYLE["bg"]
        )
        count_label.pack(side=tk.RIGHT)
        self.count_labels[category] = count_label
        
        return frame

    def _populate_list(self, category: str, items: List) -> None:
        """Populate listbox with formatted items"""
        listbox = self.listboxes[category]
        listbox.delete(0, tk.END)
        
        for text, html_info in items:
            hierarchy = self.backend.format_hierarchy(html_info)
            listbox.insert(tk.END, f"{text[:50]}...\n[{hierarchy}]\n")

    def _filter_list(self, category: str) -> None:
        """Filter list based on search text"""
        search_text = self.search_vars[category].get().lower()
        items = self.backend.selections[category]
        
        filtered = [
            (text, html_info) for text, html_info in items
            if search_text in text.lower() or 
            search_text in self.backend.format_hierarchy(html_info).lower()
        ]
        
        self._populate_list(category, filtered)
        self.count_labels[category].config(text=self.ui_text.LABELS["items_count"].format(len(filtered), len(items)))

    def select_all(self) -> None:
        """Select all items in current tab"""
        current = self.notebook.select()
        category = self.notebook.tab(current)['text'].lower()
        
        if category in self.listboxes:
            self.listboxes[category].select_set(0, tk.END)

    def focus_search(self) -> None:
        """Focus search box in current tab"""
        current = self.notebook.select()
        category = self.notebook.tab(current)['text'].lower()
        
        if category in self.search_vars:
            for child in self.notebook.winfo_children():
                if child.winfo_manager():
                    search = child.winfo_children()[0].winfo_children()[0]
                    search.focus_set()
                    search.select_range(0, tk.END)
                    break

    def handle_delete(self, event: tk.Event) -> None:
        current_tab = self.notebook.select()
        category = self.notebook.tab(current_tab)['text'].lower()
        
        if category in self.listboxes:
            self.delete_selection(self.listboxes[category], category)
        
    def delete_selection(self, listbox: tk.Listbox, category: str) -> None:
        """Delete selected items, deferring logic to the backend."""
        selection = listbox.curselection()
        if not selection:
            return
        
        count = len(selection)
        if messagebox.askyesno(
            self.ui_text.TITLES["confirm"],
            self.ui_text.MESSAGES["confirm_delete"].format(count, 's' if count > 1 else '')
        ):
            self.backend.delete_items(category, selection)
            for idx in sorted(selection, reverse=True):
                listbox.delete(idx)
            self.count_labels[category].config(text=f"Items: {listbox.size()}")

    def confirm(self):
        self.result = self.backend.selections
        self.cleanup()
        self.destroy()



class PatternTestDialog(tk.Toplevel):
    """Dialog for testing category pattern matches"""
    def __init__(self, parent, extracted_text: ExtractedText, html_map: Dict[str, HTMLInfo]):
        super().__init__(parent)
        self.title("パターン テスト ＰＡＴＴＥＲＮ ＴＥＳＴ")
        self.geometry("1000x800")
        
        # Store data
        self.extracted = extracted_text
        self.html_map = html_map
        
        # Track overlaps
        self.overlaps = self._find_overlaps()
        
        # Configure styles
        self.configure(bg=ViewerTheme.BG_COLOR)
        self.setup_ui()
        
    def setup_ui(self):
        # Split pane layout
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Text display
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)
        
        self.text_display = tk.Text(
            left_frame,
            wrap=tk.WORD,
            **ViewerTheme.TEXT_STYLE
        )
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure highlight tags
        self.text_display.tag_configure(
            "match",
            background="#05FFA1"  # Neon green
        )
        self.text_display.tag_configure(
            "overlap",
            background="#FF3F3F"  # Neon red
        )
        
        # Right side - Controls
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Category toggle buttons
        ttk.Label(
            right_frame,
            text="カテゴリー CATEGORIES",
            font=ViewerTheme.HEADER_FONT
        ).pack(pady=10)
        
        self.category_vars = {}
        for category in ["headers", "subheaders", "body", "footnotes"]:
            var = tk.BooleanVar()
            self.category_vars[category] = var
            
            count = self._get_category_count(category)
            tk.Checkbutton(
                right_frame,
                text=f"{ViewerText.LABELS[category]} ({count})",
                variable=var,
                command=lambda c=category: self.toggle_category(c),
                bg=ViewerTheme.BG_COLOR,
                fg=ViewerTheme.FG_COLOR,
                selectcolor=ViewerTheme.ACCENT_COLOR,
                activebackground=ViewerTheme.BG_COLOR,
                activeforeground=ViewerTheme.FG_COLOR
            ).pack(anchor=tk.W, padx=5, pady=2)
            
        # Stats
        stats_frame = ttk.LabelFrame(
            right_frame, 
            text="トウケイ STATISTICS",
            padding=10
        )
        stats_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.stats_label = ttk.Label(
            stats_frame,
            text=self._get_stats_text(),
            font=ViewerTheme.MONO_FONT
        )
        self.stats_label.pack()
        
        # Load initial content
        self._load_content()
        
    def _find_overlaps(self):
        """Find text that matches multiple categories"""
        overlaps = []
        all_texts = {}
        
        for category in ["headers", "subheaders", "body", "footnotes"]:
            for block in getattr(self.extracted, category):
                if block.text in all_texts:
                    overlaps.append(block.text)
                all_texts[block.text] = category
                
        return overlaps
        
    def _get_category_count(self, category):
        """Get count of matches for category"""
        return len(getattr(self.extracted, category))
        
    def _get_stats_text(self):
        """Get statistics summary"""
        total = sum(self._get_category_count(c) for c in ["headers", "subheaders", "body", "footnotes"])
        return (
            f"Total matches: {total}\n"
            f"Overlaps: {len(self.overlaps)}\n"
        )
        
    def _load_content(self):
        """Load all content into text display"""
        self.text_display.delete(1.0, tk.END)
        
        # Load content from extracted text
        for category in ["headers", "subheaders", "body", "footnotes"]:
            for block in getattr(self.extracted, category):
                start = self.text_display.index("end-1c")
                self.text_display.insert(tk.END, f"{block.text}\n\n")
                end = self.text_display.index("end-1c")
                
                # Store text range for category
                if not hasattr(self, f"{category}_ranges"):
                    setattr(self, f"{category}_ranges", [])
                getattr(self, f"{category}_ranges").append((start, end, block.text))
                
                # Mark overlaps
                if block.text in self.overlaps:
                    self.text_display.tag_add("overlap", start, end)
                    
    def toggle_category(self, category):
        """Toggle highlighting for category"""
        # Clear previous highlighting
        self.text_display.tag_remove("match", "1.0", tk.END)
        
        # Add highlighting for checked categories
        for cat, var in self.category_vars.items():
            if var.get():
                ranges = getattr(self, f"{cat}_ranges")
                for start, end, _ in ranges:
                    self.text_display.tag_add("match", start, end)



@dataclass 
class TOCExtractorText(DialogText):
    """Text constants for TOC extractor dialog"""
    TITLES = {
        "main": "ToC Extractor",
        "candidates": "Select ToC Page",
        "structure": "Review Structure",
        "entry": "Edit Entry",
        "extraction": "Extraction",
        "block_review": "Review Block",
        "rendered_text": "Text Preview",
        "html_source": "HTML Source"
    }
    
    LABELS = {
        "page": "Page {}/{}",
        "toc_title": "Table of Contents", 
        "hierarchy": "ToC Hierarchy",
        "level": "Level {}",
        "parent": "Parent Entry",
        "position": "Position",
        "move": "Move Entry"
    }
    
    BUTTONS = {
        "prev": "Previous",
        "next": "Next",
        "confirm": "Confirm Page",
        "back": "Back",
        "continue": "Continue",
        "add": "Add Entry",
        "remove": "Remove",
        "move": "Move",
        "save": "Save Changes",
        "remove_pattern": "Remove as Pattern",
        "cancel": "Cancel"
    }
    
    MESSAGES = {
        "confirm_page": "Use this page as Table of Contents?",
        "confirm_structure": "Is this ToC structure correct?",
        "confirm_remove": "Remove this entry and its children?",
        "no_parent": "Entry must have a parent",
        "invalid_position": "Invalid position",
        "confirm_remove_block": "Remove this block and create a pattern to exclude similar content?",
        "pattern_created": "Pattern created successfully"
    }

class TOCExtractorDialogUI(BaseDialog):
    """Dialog for extracting and editing table of contents"""
    
    FIRST_N_BLOCKS = 200

    def __init__(
        self, 
        parent: tk.Tk,
        epub_path: str,
        title: str,
        style: DialogStyle = None,
        ui_text: TOCExtractorText = None
    ):
        self.toc_backend = TOCExtractor(epub_path)
        self.html_backend = EPUBSelectorBackend()
        self.ui_text = ui_text or TOCExtractorText()
        
        # UI state
        self.current_page = 0
        self.candidates = []
        self.selected_entries = set()
        
        super().__init__(parent, title, style=style)
        
        # Load candidates
        self.load_candidates()
        
    def setup_ui(self):
        """Setup tabbed interface for extraction workflow"""
        # Configure ttk style
        style = ttk.Style()
        style.configure(
            "TOC.TNotebook",
            background=self.style.BG_COLOR,
            foreground=self.style.FG_COLOR
        )
        
        # Main notebook for workflow pages
        self.notebook = ttk.Notebook(self, style="TOC.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create workflow pages
        self.candidate_page = self._create_candidate_page()
        self.structure_page = self._create_structure_page()
        self.review_page = self._create_review_page()
        
        self.notebook.add(self.candidate_page, text=self.ui_text.TITLES["candidates"])
        self.notebook.add(self.structure_page, text=self.ui_text.TITLES["structure"])
        self.notebook.add(self.review_page, text=self.ui_text.TITLES["extraction"])

        # Configure hover tag
        for widget in (self.text_display, self.html_preview):
            widget.tag_configure(
                "hover",
                background=self.style.HOVER_STYLE["background"],
                foreground=self.style.HOVER_STYLE["foreground"]
            )
            widget.bind("<Motion>", self.handle_hover)
            widget.bind("<Leave>", self.handle_leave)
            widget.bind("<Double-Button-1>", self.handle_double_click)
        self.current_hover_tag: str = None
        
    def _create_candidate_page(self) -> ttk.Frame:
        """Create page for selecting TOC page"""
        frame = ttk.Frame(self.notebook)
        
        # Navigation controls
        nav_frame = ttk.Frame(frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.page_label = ttk.Label(
            nav_frame,
            text=self.ui_text.LABELS["page"].format(1, len(self.candidates))
        )
        self.page_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            nav_frame,
            text=self.ui_text.BUTTONS["prev"],
            command=self.prev_page,
            style="TOC.TButton"
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            nav_frame,
            text=self.ui_text.BUTTONS["next"], 
            command=self.next_page,
            style="TOC.TButton"
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            nav_frame,
            text=self.ui_text.BUTTONS["confirm"],
            command=self.confirm_page,
            style="TOC.TButton"  
        ).pack(side=tk.RIGHT)
        
        # HTML preview
        preview_frame = ttk.Frame(frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.html_preview = tk.Text(
            preview_frame,
            wrap=tk.WORD,
            **self.style.TEXT_STYLE
        )
        self.html_preview.pack(fill=tk.BOTH, expand=True)
        
        return frame
        
    def _create_structure_page(self) -> ttk.Frame:
        """Create page for reviewing TOC structure"""
        frame = ttk.Frame(self.notebook)
        
        # Split view
        paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # HTML preview
        preview_frame = ttk.Frame(paned)
        paned.add(preview_frame)
        
        self.text_display = tk.Text(
            preview_frame, 
            wrap=tk.WORD,
            **self.style.TEXT_STYLE
        )
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Hierarchy tree
        tree_frame = ttk.Frame(paned) 
        paned.add(tree_frame)
        
        self.tree = ttk.Treeview(
            tree_frame,
            selectmode="browse",
            style="TOC.Treeview"
        )
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Button controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            control_frame,
            text=self.ui_text.BUTTONS["add"],
            command=self.add_entry,
            style="TOC.TButton"
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            control_frame,
            text=self.ui_text.BUTTONS["remove"],
            command=self.remove_entry,
            style="TOC.TButton"  
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            control_frame,
            text=self.ui_text.BUTTONS["move"],
            command=self.move_entry,
            style="TOC.TButton"
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            control_frame,
            text=self.ui_text.BUTTONS["save"],
            command=self.save_structure,
            style="TOC.TButton"
        ).pack(side=tk.RIGHT)
        
        return frame
    
    def _create_review_page(self) -> ttk.Frame:
        """Create page for reviewing extracted text blocks"""
        frame = ttk.Frame(self.notebook)
        frame.pack(fill=tk.BOTH, expand=True)


        self.review_text = HTMLTextViewer(frame, self.style)
        self.review_text.pack(fill="both", expand=True)
        self.review_text.on_select = self._on_block_selected
        
        control_frame = ttk.Frame(frame)

        control_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        ttk.Button(
            control_frame,
            text=self.ui_text.BUTTONS["remove"],
            command=self.remove_artifact,
            style="TOC.TButton"
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            control_frame,
            text=self.ui_text.BUTTONS["save"],
            command=self.save_structure,
            style="TOC.TButton"
        ).pack(side=tk.RIGHT)
        
        return frame
    
    def handle_hover(self, event: tk.Event) -> None:
        """Handle text hover with highlighting"""
        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")
        
        try:
            bbox = widget.bbox(index)
            if not bbox:
                return

            x1, y1, width, height = bbox
            x2 = x1 + width
            y2 = y1 + height

            # Add padding to detection area
            hover_padding = 5  # Adjust for larger/smaller detection area
            mouse_x, mouse_y = event.x, event.y 

            # Check if mouse is within padded area
            if (x1 - hover_padding <= mouse_x <= x2 + hover_padding and
                y1 - hover_padding <= mouse_y <= y2 + hover_padding):
                # Get tags at current position
                tags = widget.tag_names(index)
                block_tags = [t for t in tags if t.startswith('block_')]
                
                if block_tags:
                    for tag in block_tags:
                        ranges = widget.tag_ranges(tag)
                        if ranges:
                            block_start, block_end = ranges[0], ranges[1]
                            text = widget.get(block_start, block_end).strip()
                            
                            if text and text != self.current_hover_tag:
                                self._reset_hover(widget)
                                widget.tag_add("hover", block_start, block_end)
                                widget.configure(cursor="hand2")
                                self.current_hover_tag = text
                                return
                
                self._reset_hover(widget)
                
        except tk.TclError:
            self._reset_hover(widget)

    def _reset_hover(self, widget: tk.Text) -> None:
        """Reset hover state for widget"""
        if self.current_hover_tag:
            widget.tag_remove("hover", "1.0", tk.END)
            widget.configure(cursor="arrow")
            self.current_hover_tag = None

    def handle_leave(self, event: tk.Event) -> None:
        """Reset hover when mouse leaves widget"""
        self._reset_hover(event.widget)

    def handle_double_click(self, event: tk.Event) -> None:
        """Handle text selection for entry addition"""
        widget = event.widget
        try:
            selection = widget.tag_ranges("sel")
            if selection:
                text = widget.get(*selection)
                self.add_entry_from_text(text)
        except tk.TclError:
            pass
        
    def load_candidates(self):
        """Load TOC candidates from backend"""
        self.candidates = self.toc_backend.find_toc_candidates()
        if self.candidates:

            self.show_candidate(0)
            
    def show_candidate(self, index: int):
        """Display candidate page"""
        if 0 <= index < len(self.candidates):
            self.current_page = index
            filename, content = self.candidates[index]
            
            # Update navigation
            self.page_label.config(
                text=self.ui_text.LABELS["page"].format(
                    index + 1, 
                    len(self.candidates)
                )
            )
            parsed_text = BeautifulSoup(content, "html.parser").get_text('\n')
            # Update preview
            self.html_preview.delete("1.0", tk.END)
            self.html_preview.insert("1.0", parsed_text)
            
    def prev_page(self):
        """Show previous candidate"""
        if self.current_page > 0:
            self.show_candidate(self.current_page - 1)

            
    def next_page(self):
        """Show next candidate"""
        if self.current_page < len(self.candidates) - 1:
            self.show_candidate(self.current_page + 1)
            
    def confirm_page(self):
        """Confirm TOC page selection"""
        if messagebox.askyesno(
            self.ui_text.TITLES["candidates"],
            self.ui_text.MESSAGES["confirm_page"]
        ):
            # Set page in backend
            filename, content = self.candidates[self.current_page]
            self.toc_backend.set_toc_page(content)
            
            # Extract structure
            self.toc_backend.extract_toc_structure()
            # Update tree view
            self._populate_tree()

            self.html_backend.load_html(content)

            self._populate_text_display(self.html_backend.html_map)
            
            # Switch to structure page
            self.notebook.select(1)

    def _populate_review_text(self) -> None:
        """Display extracted text blocks with diff viewing"""
        self.review_text.text.delete("1.0", "end")
        
        def build_content(entry: TOCEntry, blocks_added: int = 0, level: int = 0) -> tuple[str, int]:
            """
            Build HTML content string for entry and children.
            Returns tuple of (content, total_blocks_added)
            """
            content = []
            
            # Add header
            content.append(f"<h{level+1}>{entry.title}</h{level+1}>")
            
            # Add blocks up to limit
            remaining = self.FIRST_N_BLOCKS - blocks_added
            if remaining > 0:
                blocks_to_add = entry.html_blocks[:remaining]
                content.extend(blocks_to_add)
                blocks_added += len(blocks_to_add)
                
                # Process children if we haven't hit limit
                for child in entry.children:
                    if blocks_added < self.FIRST_N_BLOCKS:
                        child_content, blocks_added = build_content(child, blocks_added, level + 1)
                        content.append(child_content)
                    else:
                        break
                        
            return "\n".join(content), blocks_added

        # Build complete content
        complete_html = "\n".join(
            build_content(entry)[0]  # Only take content, discard final count
            for entry in self.toc_backend.toc_structure
        )
        
        # Set content once
        self.review_text.set_html_content(complete_html)

    def _on_block_selected(self, block_tag: str) -> None:
        """Handle block selection"""
        if block_tag not in self.review_text.block_map:
            return
            
        block_info = self.review_text.block_map[block_tag]
        
        if block_info['type'] == 'content':
            # Show dialog with HTML preview
            dialog = BlockReviewDialog(
                self,
                self.ui_text.TITLES["block_review"],
                block_info['text'],
                block_info['html'],
                style=self.style,
                ui_text=self.ui_text
            )
            
            if dialog.result == 'remove':
                # Create artifact pattern
                pattern = f"^{re.escape(block_info['text'])}$"
                self.toc_backend._add_artifact_pattern(pattern)
                
                # Refresh view
                self._populate_review_text()

    def _populate_text_display(self, html_map: Dict[str, dict]):
        """
        Populate the text display with each entry in html_map, tagging each
        block so the hover and click handlers can process them.
        """
        self.text_display.delete("1.0", tk.END)
        for i, (text_key, data) in enumerate(islice(html_map.items(), self.FIRST_N_BLOCKS)):
            block_start = self.text_display.index('end-1c')
            self.text_display.insert(tk.END, text_key)
            block_end = self.text_display.index('end-1c')
            block_tag = f"block_{i}"
            self.text_display.tag_add(block_tag, block_start, block_end)
            
            # Configure block tag to be below hover
            self.text_display.tag_lower(block_tag, "hover")
            self.text_display.insert(tk.END, "\n\n")

            
    def _populate_tree(self):
        """Populate tree with TOC structure"""
        self.tree.delete(*self.tree.get_children())
        
        def add_entries(entries: List[TOCEntry], parent=""):
            for i, entry in enumerate(entries):
                item_id = self.tree.insert(
                    parent,
                    "end",
                    text=entry.title,
                    values=(entry.href, entry.level)
                )
                if entry.children:
                    add_entries(entry.children, item_id)
                    
        add_entries(self.toc_backend.toc_structure)
        
    def add_entry(self):
        """Add new TOC entry"""
        # Get selected text from HTML view
        selection = self.text_display.tag_ranges("sel")
        if not selection:
            return
            
        text = self.text_display.get(*selection)
        
        # Create entry dialog
        dialog = TOCEntryDialog(
            self,
            self.ui_text.TITLES["entry"],
            text,
            self.tree,
            style=self.style,
            ui_text=self.ui_text
        )
        
        if dialog.result:
            # Update backend structure
            entry = TOCEntry(**dialog.result)
            self._update_structure(entry)
            
            # Refresh tree
            self._populate_tree()

            
    def remove_entry(self):
        """Remove selected entry"""
        selection = self.tree.selection()
        if not selection:
            return
            
        if messagebox.askyesno(
            self.ui_text.TITLES["structure"],
            self.ui_text.MESSAGES["confirm_remove"]
        ):
            # Remove from backend
            self._remove_entry(selection[0])
            
            # Update tree
            self.tree.delete(selection[0])
            
    def move_entry(self):
        """Move selected entry"""
        selection = self.tree.selection()
        if not selection:
            return
            
        # Create move dialog
        dialog = TOCMoveDialog(
            self,
            self.ui_text.TITLES["entry"],
            selection[0],
            self.tree,
            style=self.style,
            ui_text=self.ui_text
        )
        
        if dialog.result:
            # Update backend structure
            self._move_entry(selection[0], **dialog.result)
            
            # Refresh tree
            self._populate_tree()

    def add_entry_from_text(self, text: str) -> None:
        """Create new TOC entry from selected text"""
        dialog = TOCEntryDialog(
            self,
            self.ui_text.TITLES["entry"],
            text,
            self.tree,
            style=self.style,
            ui_text=self.ui_text
        )
        
        if dialog.result:
            entry = TOCEntry(**dialog.result)
            self._update_structure(entry)
            self._populate_tree()

    def remove_artifact(self) -> None:
        """Remove selected text as artifact pattern"""
        selection = self.review_text.tag_ranges("sel")
        if selection:
            text = self.review_text.get(*selection).strip()
            pattern = f"^{re.escape(text)}$"
            self.toc_backend._add_artifact_pattern(pattern)
            
            # Refresh review text
            self._populate_review_text()
            
    def save_structure(self):
        """Extract text and switch to review page"""
        if messagebox.askyesno(
            self.ui_text.TITLES["structure"],
            self.ui_text.MESSAGES["confirm_structure"]
        ):
            # Create progress dialog
            progress = ttk.Progressbar(
                self, 
                mode='indeterminate',
                length=300
            )
            progress.pack(pady=10)
            progress.start()
            
            try:
                # Extract text blocks with TOC structure
                self.update_idletasks()
                self.toc_backend.extract_text_blocks()
                
                # Populate review page
                self._populate_review_text()
                
                # Switch to review page
                self.notebook.select(2)  # Index of review page
                
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to extract text: {str(e)}"
                )
                self._handle_error(e)
                raise
                
            finally:
                self.result = None
                progress.stop()
                progress.destroy()
    def _handle_error(self, error: Exception) -> None:
        """Handle error with error dialog"""
        messagebox.showerror(
            self.ui_text.TITLES["main"],
            str(error)
        )
        log_error(error)

    def on_close(self):
        """Handle window close with confirmation"""
        if messagebox.askyesno(
            VaporwaveFormatter.format_title("Confirm Exit"),
            "Are you sure you want to exit? All progress will be lost."
        ):
            self.result = False
            self.cleanup()
            self.destroy()

    def confirm_extraction(self):
        """Final confirmation and close"""
        self.result = self.toc_backend.toc_structure
        self.cleanup()
        self.destroy()        
    
    def _update_structure(self, entry: TOCEntry):
        """Update backend TOC structure with new entry"""
        if entry.level == 0:
            self.toc_backend.toc_structure.append(entry)
        else:
            # Find parent
            def find_parent(entries: List[TOCEntry], level: int) -> Optional[TOCEntry]:
                for e in entries:
                    if e.level == level - 1:
                        return e
                    if e.children:
                        parent = find_parent(e.children, level)
                        if parent:
                            return parent
                return None
                
            parent = find_parent(self.toc_backend.toc_structure, entry.level)
            if parent:
                parent.children.append(entry)
            else:
                raise ValueError(self.ui_text.MESSAGES["no_parent"])
                
    def _remove_entry(self, item_id: str):
        """Remove entry from backend structure"""
        # Get entry info
        entry_text = self.tree.item(item_id)["text"]
        
        def remove_entry(entries: List[TOCEntry]):
            for i, entry in enumerate(entries):
                if entry.title == entry_text:
                    entries.pop(i)
                    return True
                if entry.children and remove_entry(entry.children):
                    return True
            return False
            
        remove_entry(self.toc_backend.toc_structure)
        
    def _move_entry(self, item_id: str, parent_id: str, position: int):
        """Move entry in backend structure"""
        # Get entry info
        entry_text = self.tree.item(item_id)["text"]
        
        # Find and remove entry
        def find_entry(entries: List[TOCEntry]) -> Optional[TOCEntry]:
            for i, entry in enumerate(entries):
                if entry.title == entry_text:
                    return entries.pop(i)
                if entry.children:
                    found = find_entry(entry.children)
                    if found:
                        return found
            return None
            
        entry = find_entry(self.toc_backend.toc_structure)
        if not entry:
            return
            
        # Add to new parent
        if parent_id:
            parent_text = self.tree.item(parent_id)["text"]
            
            def find_parent(entries: List[TOCEntry]) -> Optional[TOCEntry]:
                for entry in entries:
                    if entry.title == parent_text:
                        return entry
                    if entry.children:
                        parent = find_parent(entry.children)
                        if parent:
                            return parent
                return None
                
            parent = find_parent(self.toc_backend.toc_structure)
            if parent:  
                entry.level = parent.level + 1
                if 0 <= position <= len(parent.children):
                    parent.children.insert(position, entry)
                else:
                    raise ValueError(self.ui_text.MESSAGES["invalid_position"])
                    
        else:  # Move to root
            entry.level = 0
            if 0 <= position <= len(self.toc_backend.toc_structure):
                self.toc_backend.toc_structure.insert(position, entry)
            else:
                raise ValueError(self.ui_text.MESSAGES["invalid_position"])

class TOCEntryDialog(BaseDialog):
    """Dialog for creating/editing TOC entries"""
    
    def __init__(
        self, 
        parent: tk.Tk,
        title: str,
        text: str,
        tree: ttk.Treeview,
        style: DialogStyle = None,
        ui_text: TOCExtractorText = None
    ):
        self.text = text
        self.tree = tree
        self.ui_text = ui_text or TOCExtractorText()
        super().__init__(parent, title, style=style)
        
    def setup_ui(self):
        # Entry details frame
        details = ttk.LabelFrame(self, text=self.ui_text.LABELS["entry"])
        details.pack(fill=tk.X, padx=10, pady=5)
        
        # Title (from selected text)
        ttk.Label(details, text="Title:").grid(row=0, column=0, padx=5, pady=5)
        title_var = tk.StringVar(value=self.text)
        ttk.Entry(details, textvariable=title_var).grid(row=0, column=1, sticky='ew')
        
        # Parent selection
        ttk.Label(details, text=self.ui_text.LABELS["parent"]).grid(row=1, column=0, padx=5, pady=5)
        parent_var = tk.StringVar()
        parent_cb = ttk.Combobox(details, textvariable=parent_var)
        parent_cb.grid(row=1, column=1, sticky='ew')
        
        # Populate parent options
        parents = ["(Root Level)"]  # Root level option
        for item_id in self.tree.get_children():
            parents.append(self.tree.item(item_id)["text"])
        parent_cb["values"] = parents
        parent_cb.set(parents[0])
        
        # Position spinbox 
        ttk.Label(details, text=self.ui_text.LABELS["position"]).grid(row=2, column=0, padx=5, pady=5)
        position_var = tk.StringVar(value="0")
        position_sb = ttk.Spinbox(
            details,
            from_=0,
            to=100,  # Arbitrary max
            textvariable=position_var,
            width=5
        )
        position_sb.grid(row=2, column=1, sticky='w')
        
        # Update position max when parent changes
        def update_position_max(*args):
            parent_text = parent_var.get()
            if parent_text == "(Root Level)":
                position_sb["to"] = len(self.tree.get_children())
            else:
                for item_id in self.tree.get_children():
                    if self.tree.item(item_id)["text"] == parent_text:
                        position_sb["to"] = len(self.tree.get_children(item_id))
                        break
                        
        parent_var.trace_add("write", update_position_max)
        
        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame,
            text=self.ui_text.BUTTONS["confirm"],
            command=lambda: self.confirm(title_var.get(), parent_var.get(), int(position_var.get()))
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame,
            text=self.ui_text.BUTTONS["cancel"],
            command=self.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
    def confirm(self, title: str, parent: str, position: int):
        """Validate and store result"""
        # Determine level based on parent
        if parent == "(Root Level)":
            level = 0
            parent_id = ""
        else:
            # Find parent item
            for item_id in self.tree.get_children():
                if self.tree.item(item_id)["text"] == parent:
                    level = int(self.tree.item(item_id)["values"][1]) + 1
                    parent_id = item_id
                    break
                    
        self.result = {
            "title": title,
            "href": "#",  # Placeholder href
            "level": level,
            "parent_id": parent_id,
            "position": position
        }
        self.destroy()


class TOCMoveDialog(BaseDialog):
    """Dialog for moving TOC entries"""
    
    def __init__(
        self,
        parent: tk.Tk,
        title: str, 
        item_id: str,
        tree: ttk.Treeview,
        style: DialogStyle = None,
        ui_text: TOCExtractorText = None
    ):
        self.item_id = item_id
        self.tree = tree
        self.ui_text = ui_text or TOCExtractorText()
        super().__init__(parent, title, style=style)
        
    def setup_ui(self):
        # Move options frame
        options = ttk.LabelFrame(self, text=self.ui_text.LABELS["move"])
        options.pack(fill=tk.X, padx=10, pady=5)
        
        # Current item info
        current_text = self.tree.item(self.item_id)["text"]
        ttk.Label(
            options,
            text=f"Moving: {current_text}"
        ).grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # New parent selection
        ttk.Label(
            options,
            text=self.ui_text.LABELS["parent"]
        ).grid(row=1, column=0, padx=5, pady=5)
        
        parent_var = tk.StringVar()
        parent_cb = ttk.Combobox(options, textvariable=parent_var)
        parent_cb.grid(row=1, column=1, sticky='ew')
        
        # Populate parent options, excluding self and children
        def get_valid_parents(item_id):
            parents = ["(Root Level)"]
            children = set()
            
            def collect_children(node):
                for child in self.tree.get_children(node):
                    children.add(child)
                    collect_children(child)
                    
            collect_children(item_id)
            
            for node in self.tree.get_children():
                if node != item_id and node not in children:
                    parents.append(self.tree.item(node)["text"])
                    
            return parents
            
        parent_cb["values"] = get_valid_parents(self.item_id)
        parent_cb.set("(Root Level)")
        
        # Position spinbox
        ttk.Label(
            options,
            text=self.ui_text.LABELS["position"]
        ).grid(row=2, column=0, padx=5, pady=5)
        
        position_var = tk.StringVar(value="0")
        position_sb = ttk.Spinbox(
            options,
            from_=0,
            to=100,
            textvariable=position_var,
            width=5
        )
        position_sb.grid(row=2, column=1, sticky='w')
        
        # Update position max when parent changes
        def update_position_max(*args):
            parent_text = parent_var.get()
            if parent_text == "(Root Level)":
                position_sb["to"] = len(self.tree.get_children())
            else:
                for item_id in self.tree.get_children():
                    if self.tree.item(item_id)["text"] == parent_text:
                        position_sb["to"] = len(self.tree.get_children(item_id))
                        break
                        
        parent_var.trace_add("write", update_position_max)
        
        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame,
            text=self.ui_text.BUTTONS["confirm"],
            command=lambda: self.confirm(
                parent_var.get(),
                int(position_var.get())
            )
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame,
            text=self.ui_text.BUTTONS["cancel"],
            command=self.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
    def confirm(self, parent: str, position: int):
        """Store move parameters"""
        # Find parent_id if not root
        parent_id = ""
        if parent != "(Root Level)":
            for item_id in self.tree.get_children():
                if self.tree.item(item_id)["text"] == parent:
                    parent_id = item_id
                    break
                    
        self.result = {
            "parent_id": parent_id,
            "position": position
        }
        self.destroy()

class BlockReviewDialog(BaseDialog):
    """Dialog for reviewing and removing text blocks"""
    
    def __init__(self, parent, title: str, text: str, html: str, 
                 style: DialogStyle, ui_text: DialogText):
        self.text = text
        self.html = html
        self.ui_text = ui_text
        super().__init__(parent, title, style=style)
        
    def setup_ui(self):
        # Text preview
        text_frame = ttk.LabelFrame(
            self, 
            text=self.ui_text.TITLES["rendered_text"]
        )
        text_frame.pack(fill=tk.X, padx=10, pady=5)
        
        text_preview = tk.Text(text_frame, height=4, wrap=tk.WORD)
        text_preview.insert("1.0", self.text)
        text_preview.pack(fill=tk.X, padx=5, pady=5)
        
        # HTML preview  
        html_frame = ttk.LabelFrame(
            self, 
            text=self.ui_text.TITLES["html_source"]
        )
        html_frame.pack(fill=tk.X, padx=10, pady=5)
        
        html_preview = tk.Text(html_frame, height=8, wrap=tk.WORD)
        html_preview.insert("1.0", self.html)
        html_preview.pack(fill=tk.X, padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            btn_frame,
            text=self.ui_text.BUTTONS["remove_pattern"],
            command=self._remove
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            btn_frame, 
            text=self.ui_text.BUTTONS["cancel"],
            command=self.destroy
        ).pack(side=tk.RIGHT, padx=5)

    def _remove(self):
        if messagebox.askyesno(
            self.ui_text.TITLES["block_review"],
            self.ui_text.MESSAGES["confirm_remove_block"]
        ):
            self.result = 'remove'
            messagebox.showinfo(
                self.ui_text.TITLES["block_review"],
                self.ui_text.MESSAGES["pattern_created"]
            )
            self.destroy()


class HTMLTextViewer(ctk.CTkFrame):
    def __init__(self, parent, style, enable_hover=True, on_select=None, fixed_width=True, chunk_size=50):
        super().__init__(parent)
        self.style = style
        self.enable_hover = enable_hover
        self.on_select = on_select
        self.fixed_width = fixed_width
        self.chunk_size = chunk_size
        self.content_chunks = []
        self.current_chunk = 0
        self._last_width = 0
        self.original_html = ""
        self.parser = None

        self.setup_ui()
        self.configure_tags()
        self._init_parser()

    def setup_ui(self):
        # Configure frame for expansion
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create container frame
        self.container = ttk.Frame(self)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Create text widget
        default_font = ('TkDefaultFont', 11)
        self.text = tk.Text(
            self.container,
            wrap="none" if self.fixed_width else "word",
            font=default_font,
            bg=self.style.TEXT_STYLE["bg"],
            fg=self.style.TEXT_STYLE["fg"],
            insertbackground=self.style.TEXT_STYLE["fg"],
            selectbackground=self.style.ACCENT_COLOR,
            selectforeground=self.style.TEXT_STYLE["bg"],
            padx=5,
            pady=5,
            cursor="arrow"
        )
        
        # Create scrollbar
        self.scrollbar = ttk.Scrollbar(
            self.container, 
            orient="vertical", 
            command=self.text.yview
        )
        self.text.configure(yscrollcommand=self.scrollbar.set)
        
        # Grid layout
        self.text.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Event bindings
        if self.enable_hover:
            self.text.bind("<Motion>", self._on_hover)
            self.text.bind("<Leave>", self._on_leave)
        self.text.bind("<Button-1>", self._on_click)
        self.text.bind("<MouseWheel>", self._on_scroll)
        self.text.bind('<Configure>', self._on_resize)
        
        # Initialize state
        self.current_hover_tag = None
        self.block_map = {}
        
    def configure_tags(self):
        """Configure both formatting and highlighting tags"""
        # Use the explicitly configured font instead of getting from widget
        current_font = self.text['font']
        if isinstance(current_font, str):
            font_family, font_size = current_font.split()
            font_size = int(font_size)
        else:
            font_family, font_size = current_font
            
        # Text formatting tags
        self.text.tag_configure(
            "bold", 
            font=(font_family, font_size, 'bold')
        )
        self.text.tag_configure(
            "italic",
            font=(font_family, font_size, 'italic')
        )
        self.text.tag_configure(
            "underline",
            underline=1
        )
        
        # Highlighting tags
        self.text.tag_configure(
            "hover",
            background=self.style.HOVER_STYLE["background"],
            foreground=self.style.HOVER_STYLE["foreground"]
        )
        self.text.tag_configure(
            "selected",
            background=self.style.ACCENT_COLOR,  # Use a valid color code
            foreground=self.style.TEXT_STYLE["bg"]
        )
    def _init_parser(self):
        """Initialize HTML parser for text display"""
        if self.text and isinstance(self.block_map, dict):
            self.text.delete("1.0", "end")

            # Wait for widget to be drawn
            self.update_idletasks()
            
            # Get actual width or use fallback
            width = self.text.winfo_width()
            if width <= 1:
                # Use parent's width as fallback
                width = self.winfo_width()
                if width <= 1:
                    # Use reasonable default if no width available
                    width = 800

            # Calculate initial width
            char_width = self.text.tk.call('font', 'measure', self.text.cget('font'), 'x')
            chars_per_line = self.text.winfo_width() // char_width
            self._last_width = self.text.winfo_width()
            
            self.parser = HTMLTextParser(
                self.text, 
                self.block_map,
                fixed_width=self.fixed_width,
                width=chars_per_line
            )
        else:
            self.parser = None

    def set_html_content(self, html_content: str | list[str]):
        """Parse and display HTML content with chunked loading"""
        self.original_html = html_content
        self.text.delete("1.0", "end")
        self._init_parser()
        
        # Split content into chunks
        if isinstance(html_content, str):
            soup = BeautifulSoup(html_content, 'html.parser')
            blocks = soup.find_all(['p', 'div', 'h1', 'h2', 'h3'])
        elif isinstance(html_content, list):
            blocks = html_content
        else:
            raise TypeError("Invalid content type")
        
        self.content_chunks = []
        current_chunk = []
        current_size = 0
        
        for block in blocks:
            if current_size >= self.chunk_size:
                self.content_chunks.append(''.join(map(str, current_chunk)))
                current_chunk = []
                current_size = 0
            current_chunk.append(block)
            current_size += len(str(block))
            
        if current_chunk:
            self.content_chunks.append(''.join(map(str, current_chunk)))
            
        # Load first chunk
        if self.content_chunks:
            self._load_chunk(0)

    def _load_chunk(self, index: int):
        """Load a specific content chunk"""
        if 0 <= index < len(self.content_chunks):
            self.current_chunk = index
            chunk_content = self.content_chunks[index]
            self.parser.feed(chunk_content)
    
    def _on_resize(self, event):
        """Handle window resize by updating text wrapping"""
        # Only process if width actually changed
        new_width = self.text.winfo_width()
        if new_width != self._last_width:
            self._last_width = new_width
            # Convert pixels to approx char width
            char_width = self.text.tk.call('font', 'measure', self.text.cget('font'), 'x')
            chars_per_line = new_width // char_width
            
            # Recreate parser with new width
            if self.parser:
                self.text.delete("1.0", "end")
                self.block_map.clear()
                self.parser = HTMLTextParser(
                    self.text,
                    self.block_map,
                    fixed_width=self.fixed_width,
                    width=chars_per_line
                )
                # Reparse content with new width
                self.parser.feed(self.original_html)
            
    def _on_scroll(self, event):
        """Handle scroll events for lazy loading"""
        # Calculate if we're near the end
        visible_end = float(self.text.index("@0,%d" % self.text.winfo_height()))
        total_lines = float(self.text.index("end-1c").split('.')[0])
        
        if visible_end >= total_lines * 0.8 and self.current_chunk < len(self.content_chunks) - 1:
            self._load_chunk(self.current_chunk + 1)

    def add_html_content(self, html_content: str):
        """Parse and append HTML content with block tracking"""
        self.original_html += html_content
        if self.parser is None:
            self.parser = self._init_parser()
        self.parser.feed(html_content)

    def _on_hover(self, event):
        """Handle hover highlighting if enabled"""
        if not self.enable_hover:
            return
            
        # Get index at mouse position    
        index = self.text.index(f"@{event.x},{event.y}")
        
        # Clear previous hover state
        self.text.tag_remove("hover", "1.0", "end")
        self.text.configure(cursor="arrow")
        self.current_hover_tag = None
        
        # Find block tags at current position
        tags = [tag for tag in self.text.tag_names(index) if tag.startswith("block_")]
        
        if tags:
            block_tag = tags[0]  # Use first block tag found
            # Get full range of the block
            ranges = self.text.tag_ranges(block_tag)
            if ranges and len(ranges) >= 2:
                start, end = ranges[0], ranges[1]

                # Add hover highlight to entire block

                
                self.text.tag_add("hover", start, end)
                self.text.configure(cursor="hand2")
                self.current_hover_tag = block_tag
                
    def _on_leave(self, event):
        """Clear hover highlighting"""
        if self.current_hover_tag:
            self.text.tag_remove("hover", "1.0", "end")
            self.text.configure(cursor="arrow")
            self.current_hover_tag = None

    def _on_click(self, event):
        """Handle block selection and callback"""
        index = self.text.index(f"@{event.x},{event.y}")
        self.text.tag_remove("selected", "1.0", "end")
        
        for tag in self.text.tag_names(index):
            if tag.startswith("block_"):
                ranges = self.text.tag_ranges(tag)
                start, end = str(ranges[0]), str(ranges[1])
                self.text.tag_add("selected", start, end)
                
                # Call selection handler if provided
                if self.on_select and tag in self.block_map:
                    self.on_select(self.block_map[tag])
                break
@dataclass
class HTMLBlock:
    """Dataclass for storing HTML block info"""
    block_id: int
    text_start: str | None 
    text_end: str | None
    content: str
    class_: str = ""
    id: str = ""
    parent: HTMLBlock = None
    children: List[HTMLBlock] = field(default_factory=list)

class HTMLTextParser(HTMLParser):
    def __init__(self, 
                 text_widget, 
                 block_map, 
                 fixed_width: bool = True,
                 width: float = 80,
                 block_tags: List[str] = ['p', 'h1', 'h2', 'h3']):
        super().__init__()
        self.text_widget = text_widget
        self.block_map = block_map
        self.fixed_width = fixed_width
        self.width = width
        self.current_tags: dict[int, str] = {} # tag depth and font tag
        self.block_id = len(block_map)
        self.current_block: HTMLBlock = None
        self.current_line = []
        self.current_line_buffer = []  # Store (text, tags) tuples
        self.current_line_length = 0
        self.block_tags = block_tags
        
        self.tag_map = {
            'b': 'bold',
            'strong': 'bold',
            'i': 'italic', 
            'em': 'italic',
            'u': 'underline',
            'h1': 'bold',
            'h2': 'bold',
            'h3': 'bold'
        }
        self.block_tags = ['p', 'h1', 'h2', 'h3']
        self.font_classes = ['italic', 'bold']
        self.tag_depth = 0
        
    def handle_starttag(self, tag, attrs):

        # Handle formatting tags
        attrs = dict(attrs)

        if tag in self.tag_map:
            self.current_tags[self.tag_depth] = self.tag_map[tag]

        elif attrs.get('class', '') in self.font_classes:
            self.current_tags[self.tag_depth] = attrs['class']
            
        # Start new paragraph block
        if tag in self.block_tags and self.current_block is None:
            self.current_block = HTMLBlock(**{
                'class_': attrs.get('class', ''),
                'id': attrs.get('id', ''),
                'block_id': self.block_id,
                'text_start': None,
                'text_end': None,
                'content': ''
            })

        if self.current_block:
            self.tag_depth += 1

            
    def handle_endtag(self, tag):
        # Handle formatting tags
        if self.current_block:
            self.tag_depth -= 1

        if self.current_tags.get(self.tag_depth, None):
            self.current_tags.pop(self.tag_depth)
            
        # Handle block end
        if tag in self.block_tags and self.current_block and self.current_block.text_start and self.current_block.text_end: 
            # Add block tag spanning the content
            block_tag = f"block_{self.block_id}"
            
            # Force insert any buffered content
            if self.current_line_buffer:
                self._insert_buffered_line(add_newline=False)
                
            # Always add newline at block end
            self.text_widget.insert("end", "\n\n")
            
            # Set block end position before newlines
            self.current_block.text_end = self.text_widget.index("end-2c")
            
            # Add block tag
            self.text_widget.tag_add(block_tag, 
                self.current_block.text_start, 
                self.current_block.text_end
            )
            
            self.block_map[block_tag] = self.current_block
            self.block_id += 1
            self.current_block = None
            self.current_line_length = 0
            self.current_line_buffer = []      
            
    def _insert_buffered_line(self, add_newline=False):
        """Insert buffered line content with proper formatting"""
        if not self.current_line_buffer:
            return
            
        # Insert each segment with its tags
        for text, tags in self.current_line_buffer:
            self.text_widget.insert("end", text, tags if tags else "")
        
        if self.current_block and self.current_block.text_start:
            self.current_block.text_end = self.text_widget.index("end-1c")

        if add_newline:
            self.text_widget.insert("end", "\n")
            
        self.current_line_buffer = []
        self.current_line_length = 0
        
    def handle_data(self, data):
        # Track block start
        if self.current_block and not self.current_block.text_start and data != '\n':
            self.current_block.text_start = self.text_widget.index("end-1c")

        # Get current formatting tags
        tags = tuple(self.current_tags.values())
        
        if self.fixed_width:
            words = data.split()
            for word in words:
                if word == '\n':
                    self._insert_buffered_line(add_newline=True)
                    continue
                word_length = len(word)
                # Add space if not first word
                space_length = 1 if self.current_line_length > 0 else 0
                
                if self.current_line_length + word_length + space_length <= self.width:
                    # Add space between words
                    if self.current_line_length > 0:
                        self.current_line_buffer.append((" ", tags))
                        self.current_line_length += 1
                    # Add word
                    self.current_line_buffer.append((word, tags))
                    self.current_line_length += word_length
                else:
                    # Line is full - insert it and start new line
                    self._insert_buffered_line(add_newline=True)
                    # Start new line with current word
                    self.current_line_buffer.append((word, tags))
                    self.current_line_length = word_length
        else:
            # For non-fixed width, just buffer the text
            self.current_line_buffer.append((data, tags))
            
        # # Track block content and end
        # if self.current_block and data != '\n':
        #     self.current_block.content += data
        #     self.current_block.text_end = self.text_widget.index("end-1c")
        

if __name__ == "__main__":
    import tkinter as tk
    from epubkit.dialogueUI import TOCExtractorDialogUI, TOCExtractorText, VaporwaveFormatter
    from epubkit.viewer import ViewerText, ViewerTheme

    root = tk.Tk()
    root.title("Table of Contents")
    root.geometry("1200x1000")

    epub_dir = Path("./epubkit/resources/epubs").resolve()
    epub_files = list(epub_dir.glob("*.epub"))
    index = 1
    print(f'Found {len(epub_files)} epub files')
    print(f'Now processing {epub_files[index]}')

    ui_text = VaporwaveFormatter.format_dialog_text(TOCExtractorText)

    dialogue = TOCExtractorDialogUI(
                    parent=root,
                    epub_path=epub_files[index],
                    title=ViewerText.TITLES["toc"],
                    style=ViewerTheme,
                    ui_text=ui_text
                )
    dialogue.lift()
    dialogue.update_idletasks()

    root.withdraw()

    root.wait_window(dialogue)

    toc = dialogue.result

    root.mainloop()

