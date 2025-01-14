from itertools import islice
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from typing import Dict, List, Optional, TypedDict, get_args

from bs4 import BeautifulSoup
from tkhtmlview import HTMLLabel, HTMLScrolledText
from epubkit.parser import EPUBSelectorBackend, ImmutableTagInfo, PatternReviewBackend, CategoryType, SelectionReviewBackend, TOCEntry, TOCExtractor

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

    def on_close(self):
        """Handle window close"""
        self.cleanup()
        self.destroy()

    def destroy(self):
        """Ensure cleanup on destroy"""
        self.cleanup()
        super().destroy()

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

