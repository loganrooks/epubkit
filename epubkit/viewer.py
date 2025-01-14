from __future__ import annotations
from datetime import datetime
import logging
import sys
import threading
import tkinter as tk
from tkinter import PhotoImage, ttk, messagebox
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, replace
import json
from epubkit.debug import filter_traceback, setup_logging
from epubkit.dialogueUI import DialogText, EPUBTagSelectorUI, DialogStyle, PatternReviewDialog, SelectionReviewDialogUI, TOCExtractorDialogUI, TOCExtractorText, VaporwaveFormatter
from epubkit.parser import  ExtractedText, HTMLCategoryExtractor, HTMLInfo, ImmutableTagInfo, TOCEntry, TOCExtractor, extract_categorized_text, CategoryDict
from epubkit.search import SearchResult, SemanticSearch
from PIL import Image, ImageTk
from tkinter import filedialog
import random
import pygame
from mutagen.mp3 import MP3
import time
import shutil

DEBUG = True

DEBUG_SELECTIONS = {'headers': ["""<p id="filepos41583" class="calibre_8">
                        <span class="calibre11"><span class="bold">
                            I: THE NECESSITY, STRUCTURE, AND PRIORITY OF THE QUESTION OF BEING
                        </span></span>
                    </p>"""],
            'subheaders': ["""<p id="filepos41760" class="calibre_10">
            <span class="calibre11"><span class="bold"><span class="italic">
                <a><span>1. The Necessity for Explicitly Restating the Question of Being</span></a>
            </span></span></span>
        </p>"""],
            'body': ["""
                    <p class="calibre_6">
                        <span class="calibre12" id="span1">
                            <span data-custom="value">
                                <span class="calibre13">Content</span>
                            </span>
                        </span>
                    </p>
                    """,
                    """<p class="calibre_6">
                        <span class="calibre12">
                            <span>
                                <span class="calibre13">
                                    MORE than thirty years have passed since 
                                    <span class="italic">Being and Time</span>
                                    first appeared...
                                </span>
                            </span>
                        </span>
                    </p>
                    """,
                    """<p class="calibre_6">
                        <span class="calibre12"><span><span class="calibre13">
                            Text before <span class="italic">Being and Time</span><span></span>
                            continues after with more content
                        </span></span></span>
                    </p>
                    """],
            'footnotes': ['<p class="calibre_6"><span class="calibre9"><span>1.<span class="italic"> ‘…als thematische Frage wirklicher Untersuchung’.</span></span>When Heidegger speaks of a question as ‘thematisch’, he thinks of it as one which is taken seriously and studied in a systematic manner. While we shall often translate this adjective by its cognate, ‘thematic’, we may sometimes find it convenient to choose more flexible expressions involving the word ‘theme’. (Heidegger gives a fuller discussion on H. 363.)</span></p>',
                          '<p class="calibre_6"><span class="calibre9">4. ‘…als des möglichen Horizontes eines jeden Seinsverständnisses überhaupt…’ Throughout this work the word ‘horizon’ is used with a connotation somewhat different from that to which the English-speaking reader is likely to be accustomed. We tend to think of a horizon as something which we may widen or extend or go beyond; Heidegger, however, seems to think of it rather as something which we can neither widen nor go beyond, but which provides the limits for certain intellectual activities performed ‘within’ it.</span></p>',
                          '<p class="calibre_6"><span class="calibre9">1. While we shall ordinarily reserve the word ‘falling’ for ‘Verfallen’ (see our note 2, H. 21 above), in this sentence it represents first ‘Verfallen’ and then ‘Fallen’, the usual German word for ‘falling’. ‘Fallen’ and ‘Verfallen’ are by no means strictly synonymous; the latter generally has the further connotation of ‘decay’ or ‘deterioration’, though Heidegger will take pains to point out that in his own usage it ‘does not express any negative evaluation’. See Section 38 below. </span></p>'],
            'toc': []
        }

logger = setup_logging()

class ViewerTheme:
    """Vaporwave theme configuration"""
    # Colors
    BG_COLOR = "#330066"  # Deep purple
    FG_COLOR = "#FF71CE"  # Neon pink
    ACCENT_COLOR = "#B967FF"  # Bright purple
    SECONDARY_COLOR = "#01CDFE"  # Cyan
    SUCCESS_COLOR = "#05FFA1"  # Neon green
    ERROR_COLOR = "#FF3F3F"  # Bright red
    
    # Fonts
    MAIN_FONT = ("VCR OSD Mono", 12)  # Retro font
    BOLD_FONT = ("VCR OSD Mono", 12, "bold")
    HEADER_FONT = ("VCR OSD Mono", 14, "bold")
    MONO_FONT = ("VCR OSD Mono", 12)
    SUBHEADER_FONT = ("VCR OSD Mono", 12, "italic")

    JP_FONT = ("Noto Sans JP", 12)
    BOLD_JP_FONT = ("Noto Sans JP", 12, "bold")
    JP_HEADER_FONT = ("Noto Sans JP", 14, "bold")
    VAPORWAVE_FONT = ("VCR OSD Mono", 12)
    
    # Styles
    BUTTON_STYLE = {
        "font": JP_FONT,
        "bg": ACCENT_COLOR,
        "fg": BG_COLOR,
        "activebackground": SECONDARY_COLOR,
        "activeforeground": FG_COLOR,
        "relief": tk.RAISED,
        "borderwidth": 2,
        "padx": 10,
        "pady": 5
    }

    HOVER_STYLE = {
        "background": SECONDARY_COLOR,
        "foreground": ACCENT_COLOR,
        "font": BOLD_JP_FONT
    }

    MEDIA_PLAYER_BUTTON_STYLE = {
        "font": JP_FONT,
        "bg": FG_COLOR,
        "fg": SECONDARY_COLOR,
        "activebackground": ACCENT_COLOR,
        "activeforeground": BG_COLOR,
        "relief": tk.RAISED,
        "borderwidth": 2,
        "padx": 10,
        "pady": 5
    }
    TOOL_BAR_BUTTON_STYLE = {
        "font": JP_FONT,
        "fg": BG_COLOR,
        "activebackground": SECONDARY_COLOR,
        "activeforeground": FG_COLOR,
        "relief": tk.RAISED,
        "borderwidth": 2,
        "padx": 10,
        "pady": 5
    }

    MENU_BAR_BUTTON_STYLE = {
        "font": JP_FONT,
        "fg": SECONDARY_COLOR,
        "activebackground": ACCENT_COLOR,
        "activeforeground": BG_COLOR,
        "relief": tk.RAISED,
        "borderwidth": 2,
        "padx": 10,
        "pady": 5
    }

    
    ENTRY_STYLE = {
        "font": MONO_FONT,
        "bg": FG_COLOR,
        "fg": BG_COLOR,
        "insertbackground": BG_COLOR,
        "relief": tk.FLAT,
        "borderwidth": 1
    }
    
    TEXT_STYLE = {
        "font": JP_FONT,
        "bg": BG_COLOR,
        "fg": FG_COLOR,
        "insertbackground": FG_COLOR,
        "relief": tk.FLAT,
        "padx": 10,
        "pady": 5,
        "spacing1": 5,  # Spacing before paragraphs
        "spacing2": 2,  # Spacing between paragraphs
        "spacing3": 5   # Spacing after paragraphs
    }

    DOCUMENT_STYLE = {
        "font": MAIN_FONT,
        "bg": BG_COLOR,
        "fg": FG_COLOR,
        "insertbackground": FG_COLOR,
        "wrap": tk.WORD,
        "padx": 40,
        "pady": 20,
        "spacing1": 10,
        "spacing2": 2,
        "spacing3": 10
    }

class VaporwaveStyle(DialogStyle):
    # Colors
    BG_COLOR = "#330066"
    FG_COLOR = "#FF00FF"
    SECONDARY_COLOR = "#00FFFF" 
    ACCENT_COLOR = "#0099FF"
    
    # Fonts
    JP_FONT = ("MS Gothic", 14)
    BOLD_FONT = ("MS Gothic", 16, "bold")
    
    TEXT_STYLE = {
        "font": JP_FONT,
        "bg": BG_COLOR,
        "fg": FG_COLOR,
        "insertbackground": SECONDARY_COLOR,
        "selectbackground": ACCENT_COLOR,
        "relief": "solid",
        "borderwidth": 1,
        "highlightthickness": 1,
        "highlightcolor": SECONDARY_COLOR,
        "highlightbackground": FG_COLOR
    }
    
    BUTTON_STYLE = {
        "font": BOLD_FONT,
        "bg": BG_COLOR,
        "fg": SECONDARY_COLOR,
        "activebackground": ACCENT_COLOR,
        "activeforeground": BG_COLOR,
        "relief": "raised",
        "borderwidth": 2,
        "padx": 15,
        "pady": 8,
        "cursor": "hand2"
    }

@dataclass
class ViewerText(DialogText):
    """Vaporwave katakana text constants"""
    TITLES = {
        # Main UI
        "main": "テキストサーチャー  ＴＥＸＴ  ＳＥＡＲＣＨＥＲ",
        "options": "セッテイ  ＯＰＴＩＯＮＳ",
        "epub": "イーパブ  セレクター  ＥＰＵＢ",
        "review": "レビュー  ＲＥＶＩＥＷ",
        "progress": "ショリチュウ  ＰＲＯＧＲＥＳＳ",
        "error": "エラー  ＥＲＲＯＲ",
        "extract": "エ ゥ ツ ラ ク ツ  Ｅ Ｘ Ｔ Ｒ Ａ Ｃ Ｔ",
        "toc": "コンテンツ  ＴＯＣ",
        "embedding": "エンベッディング  ＥＭＢＥＤＤＩＮＧ",
        
        # Music Player
        "music": "オンガク  プレーヤー  ＭＵＳＩＣ  ＰＬＡＹＥＲ",
        "playlist": "プレイリスト  ＰＬＡＹＬＩＳＴ",
        
        # Dialogs
        "pattern_review": "パターン  レビュー",
        "main_review": "レビュー  ＲＥＶＩＥＷ",

        "collection": "コレクション",
        "search": "ケンサク  ＳＥＡＲＣＨ"
    }
    
    BUTTONS = {
        # Main Controls
        "load": "ロード  ＬＯＡＤ",
        "save": "セーブ  ＳＡＶＥ",
        "confirm": "カクニン  ＣＯＮＦＩＲＭ",
        "cancel": "キャンセル  ＣＡＮＣＥＬ",
        "delete": "サクジョ  ＤＥＬＥＴＥ",
        "done": "ワンチュウ  ＤＯＮＥ",
        "test": "テスト  ＴＥＳＴ",
        
        # Music Controls
        "play": "サイセイ ▶",
        "pause": "ポーズ  ||",
        "stop": "テイシ  ■",
        "next": "ツギヘ  ≫",
        "prev": "マエヘ  ≪",
        "music_player": "オンガク  ＭＵＳＩＣ",
        
        # Category Selection
        "headers": "ヘッダー  ＨＥＡＤＥＲＳ",
        "subheaders": "サブヘッダー  ＳＵＢＨＥＡＤＥＲＳ",
        "body": "ホンブン  ＢＯＤＹ",
        "footnotes": "チュウキ  ＦＯＯＴＮＯＴＥＳ",
        "toc": "コンテンツ  ＴＯＣ"
    }
    
    LABELS = {
        # Music Player
        "no_song": "ムジック ナシ  ＮＯ  ＳＯＮＧ",
        "time": "ジカン  ＴＩＭＥ",
        
        # Categories
        "headers": "ヘッダー  ＨＥＡＤＥＲＳ",
        "subheaders": "サブヘッダー  ＳＵＢＨＥＡＤＥＲＳ",
        "toc": "コンテンツ  ＴＯＣ",
        "body": "ホンブン  ＢＯＤＹ",
        "footnotes": "チュウキ  ＦＯＯＴＮＯＴＥＳ",
        
        # UI Elements
        "search": "ケンサク  ＳＥＡＲＣＨ",
        "results": "ケッカ  ＲＥＳＵＬＴＳ",
        "collections": "コレクション  ＣＯＬＬＥＣＴＩＯＮＳ",
        "embeddings": "エンベッディング  ＥＭＢＥＤＤＩＮＧＳ",
        "index": "インデックス  ＩＮＤＥＸ",
        "select": "センタク  ＳＥＬＥＣＴ",
        "processing": "ショリチュウ  ＰＲＯＣＥＳＳＩＮＧ",
        "extracting": "エ ゥ ツ ラ ク ツ  ＥＸＴＲＡＣＴ I N G",
        "pattern": "パターン  ＰＡＴＴＥＲＮ",
        "matches": "マッチ  ＭＡＴＣＨＥＳ",
        "conflicts": "コンフリクト  ＣＯＮＦＬＩＣＴＳ",
        "false_positives": "ギマッチ  ＦＡＬＳＥ ＰＯＳＩＴＩＶＥＳ",
        "total_items": "アイテムゴウケイ  ＴＯＴＡＬ: {}",
        "items_count": "アイテム  ＩＴＥＭＳ: {}/{}",
        "search": "ケンサク  ＳＥＡＲＣＨ...",
    }
    
    MESSAGES = {
        "load_error": "エラー：ロードデキマセン  ＬＯＡＤ  ＥＲＲＯＲ",
        "process_error": "エラー：ショリチュウデキマセン  ＰＲＯＣＥＳＳ  ＥＲＲＯＲ",
        "traceback": "トレースバック  ＴＲＡＣＥＢＡＣＫ",
        "save_success": "セーブ  カンリョウ  ＳＡＶＥ  ＣＯＭＰＬＥＴＥ",
        "processing": "ショリチュウ...  ＰＲＯＣＥＳＳＩＮＧ...",
        "no_selection": "センタクガ アリマセン  ＮＯ  ＳＥＬＥＣＴＩＯＮ",
        "confirm_delete": "サクジョ カクニン {} アイテム{}?",
        "error_general": "エラー：{}"
    }


class VaporwaveDesktop(tk.Toplevel):
    """Desktop background window with animated GIFs"""
    def __init__(self):
        super().__init__()
        self.gifs = []
        self.gif_labels = []
        self.current_frame = 0
        self.images = {}  # Store image references
        
        # Configure window
        self.attributes('-fullscreen', True)
        self.overrideredirect(True)  # Remove window decorations
        self.attributes('-topmost', False)  # Stay behind other windows
        
        self.setup_desktop()
        
    def setup_desktop(self):
        # Set up background
        bg_path = Path("resources/desktops/windows-xp.jpg")
        bg_image = Image.open(bg_path)
        self.images['bg'] = ImageTk.PhotoImage(bg_image)
        
        bg_label = tk.Label(self, image=self.images['bg'])
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Load GIFs
        gif_paths = glob.glob("resources/gifs/*.gif")
        for i, path in enumerate(gif_paths):
            gif = Image.open(path)
            frames = []
            try:
                while True:
                    frame = ImageTk.PhotoImage(gif.copy())
                    frames.append(frame)
                    self.images[f'gif_{i}_frame_{len(frames)}'] = frame
                    gif.seek(len(frames))
            except EOFError:
                pass
            self.gifs.append(frames)
            
            label = tk.Label(self)
            label.bind("<Button-1>", self.start_drag)
            label.bind("<B1-Motion>", self.drag)
            self.gif_labels.append(label)
        
        self.cycle_gifs()
        self.animate_gifs()
        
    def start_drag(self, event):
        widget = event.widget
        widget._drag_start_x = event.x
        widget._drag_start_y = event.y
        
    def drag(self, event):
        widget = event.widget
        x = widget.winfo_x() + event.x - widget._drag_start_x
        y = widget.winfo_y() + event.winfo_y() - widget._drag_start_y
        widget.place(x=x, y=y)
        
    def cycle_gifs(self):
        # Randomly show/hide GIFs every minute
        for label in self.gif_labels:
            if random.random() > 0.5:
                x = random.randint(0, self._root().winfo_width() - 200)
                y = random.randint(0, self._root().winfo_height() - 200)
                label.place(x=x, y=y)
            else:
                label.place_forget()
        self._root().after(60000, self.cycle_gifs)
        
    def animate_gifs(self):
        # Animate visible GIFs
        for i, label in enumerate(self.gif_labels):
            if label.winfo_viewable():
                frames = self.gifs[i]
                label.configure(image=frames[self.current_frame % len(frames)])
        self.current_frame += 1
        self._root().after(100, self.animate_gifs)

class SearchToolbar(ttk.Frame):
    """Enhanced search toolbar with live suggestions"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.setup_ui()
        
    def setup_ui(self):
        # Search box with icon
        search_frame = ttk.Frame(self)
        search_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(
            search_frame,
            textvariable=self.search_var,
            width=40,
            font=('MS Gothic', 12)
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Category filter
        self.category_var = tk.StringVar(value="all")
        self.category_menu = tk.OptionMenu(
            self,
            self.category_var,
            "Ａｌｌ",  # Fullwidth text
            "Ｈｅａｄｅｒｓ", 
            "Ｂｏｄｙ",
            "Ｆｏｏｔｎｏｔｅｓ"
        )
        self.category_menu.pack(side=tk.LEFT, padx=5)

        self.category_menu.config(
            width=15,
            font=('MS Gothic', 10),
            foreground='#ff71ce',  # Vaporwave pink
            background='#080428'   # Dark purple
        )
        
        # Search button
        self.search_button = tk.Button(
            self,
            text="Search",
            width=100,
            height=35,
            command=self.on_search
        )
        self.search_button.pack(side=tk.LEFT, padx=5)
        
        # Bind events
        self.search_entry.bind("<Return>", lambda e: self.on_search())
        self.search_var.trace_add("write", self.on_search_change)
    
    def on_search(self):
        """Override this method to handle search"""
        pass
        
    def on_search_change(self, *args):
        """Handle live search updates"""
        pass

class ResultsList(ttk.Frame):
    """Scrollable list of search results with previews"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.setup_ui()
        
    def setup_ui(self):
        # Results count header
        self.header = ttk.Label(
            self,
            text="Results",
            font=ViewerTheme.HEADER_FONT
        )
        self.header.pack(fill=tk.X, padx=10, pady=5)
        
        # Scrollable canvas for results
        self.canvas = tk.Canvas(
            self,
            bg=ViewerTheme.BG_COLOR,
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            self,
            orient=tk.VERTICAL,
            command=self.canvas.yview
        )
        
        self.results_frame = ttk.Frame(self.canvas)
        self.results_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.results_frame, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def add_result(self, result: SearchResult):
        """Add a search result with preview"""
        frame = ttk.Frame(self.results_frame)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Header with similarity score
        header = ttk.Label(
            frame,
            text=f"Match: {result.similarity:.2f}",
            font=ViewerTheme.HEADER_FONT
        )
        header.pack(fill=tk.X)
        
        # Preview text
        preview = ttk.Label(
            frame,
            text=result.text[:200] + "...",
            wraplength=400
        )
        preview.pack(fill=tk.X)
        
        # Action buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(
            button_frame,
            text="View in Context",
            width=120,
            command=lambda: self.on_view_context(result)
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame,
            text="Add to Collection",
            width=120,
            command=lambda: self.on_add_to_collection(result)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
    
    def clear(self):
        """Clear all results"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
    def on_view_context(self, result: SearchResult):
        """Override to handle context view"""
        pass
        
    def on_add_to_collection(self, result: SearchResult):
        """Override to handle collection add"""
        pass

class CollectionPanel(ttk.Frame):
    """Panel for managing quote collections"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.collections: Dict[str, List[SearchResult]] = {}
        self.setup_ui()
        
    def setup_ui(self):
        # Controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(
            control_frame,
            text="New Collection",
            width=120,
            command=self.new_collection
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="Export",
            width=80,
            command=self.export_collections
        ).pack(side=tk.LEFT, padx=5)
        
        # Collections notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
    def new_collection(self):
        """Create new collection"""
        name = tk.simpledialog.askstring(
            "New Collection",
            "Enter collection name:",
            parent=self
        )
        if name:
            self.add_collection(name)
    
    def add_collection(self, name: str):
        """Add new collection tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=name)
        
        # Quote list with context preview
        self.collections[name] = []
        
        listbox = tk.Listbox(
            frame,
            font=ViewerTheme.MAIN_FONT,
            bg=ViewerTheme.BG_COLOR,
            fg=ViewerTheme.FG_COLOR,
            selectmode=tk.SINGLE
        )
        listbox.pack(fill=tk.BOTH, expand=True)
        
        # Right-click menu
        menu = tk.Menu(listbox, tearoff=0)
        menu.add_command(
            label="Remove",
            command=lambda: self.remove_quote(name, listbox.curselection()[0])
        )
        menu.add_command(
            label="View Context",
            command=lambda: self.view_quote(name, listbox.curselection()[0])
        )
        
        listbox.bind("<Button-3>", lambda e: menu.post(e.x_root, e.y_root))
        
    def add_quote(self, collection: str, result: SearchResult):
        """Add quote to collection"""
        if collection not in self.collections:
            return
            
        self.collections[collection].append(result)
        self._update_collection_display(collection)
        
    def remove_quote(self, collection: str, index: int):
        """Remove quote from collection"""
        if collection not in self.collections:
            return
            
        del self.collections[collection][index]
        self._update_collection_display(collection)
        
    def _update_collection_display(self, collection: str):
        """Update collection listbox"""
        tab_id = None
        for id, name in self.notebook.tabs().items():
            if self.notebook.tab(id, "text") == collection:
                tab_id = id
                break
                
        if not tab_id:
            return
            
        frame = self.notebook.nametowidget(tab_id)
        listbox = frame.winfo_children()[0]
        listbox.delete(0, tk.END)
        
        for quote in self.collections[collection]:
            preview = quote.text[:100] + "..."
            listbox.insert(tk.END, preview)
            
    def export_collections(self):
        """Export collections to JSON"""
        path = tk.filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if path:
            data = {
                name: [quote.__dict__ for quote in quotes]
                for name, quotes in self.collections.items()
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

class RetroMusicPlayer(tk.Toplevel):
    """Retro-styled music player window"""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("﻿ＲＥＴＲＯ ＰＬＡＹＥＲ")
        self.geometry("400x300")
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        self.current_song: Optional[str] = None
        self.is_playing = False
        self.songs: List[str] = []
        self.load_songs()
        
        # Make window draggable
        self.bind("<Button-1>", self.start_drag)
        self.bind("<B1-Motion>", self.drag)
        
        self.setup_ui()
        
    def load_songs(self):
        """Load MP3 files from resources/music"""
        self.songs = list(Path("resources/music").glob("*.mp3"))
        
    def setup_ui(self):
        self.configure(bg=ViewerTheme.BG_COLOR)
        
        # Song info display
        self.info_frame = tk.Frame(
            self,
            bg=ViewerTheme.BG_COLOR,
            height=100
        )
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.song_label = tk.Label(
            self.info_frame,
            text="No song playing",
            font=ViewerTheme.HEADER_FONT,
            fg=ViewerTheme.FG_COLOR,
            bg=ViewerTheme.BG_COLOR
        )
        self.song_label.pack(pady=5)
        
        self.time_label = tk.Label(
            self.info_frame,
            text="00:00 / 00:00",
            font=ViewerTheme.MONO_FONT,
            fg=ViewerTheme.SECONDARY_COLOR,
            bg=ViewerTheme.BG_COLOR
        )
        self.time_label.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self,
            mode='determinate',
            length=350
        )
        self.progress.pack(pady=10)
        
        # Control buttons frame
        controls = tk.Frame(self, bg=ViewerTheme.BG_COLOR)
        controls.pack(pady=10)
        
        button_style = {
            "width": 60,
            "height": 30,
            "fg_color": ViewerTheme.ACCENT_COLOR,
            "hover_color": ViewerTheme.SECONDARY_COLOR
        }
        
        self.prev_btn = tk.Button(
            controls,
            text="⏮",
            command=self.previous_song,
            **button_style
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = tk.Button(
            controls,
            text="▶",
            command=self.play_pause,
            **button_style
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            controls,
            text="⏹",
            command=self.stop,
            **button_style
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(
            controls,
            text="⏭",
            command=self.next_song,
            **button_style
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Playlist button
        self.playlist_btn = tk.Button(
            self,
            text="Playlist",
            command=self.show_playlist,
            width=100
        )
        self.playlist_btn.pack(pady=10)
        
        # Update timer
        self.update_time()
        
    def start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        
    def drag(self, event):
        x = self.winfo_x() + event.x - self._drag_start_x
        y = self.winfo_y() + event.y - self._drag_start_y
        self.geometry(f"+{x}+{y}")
        
    def play_pause(self):
        if not self.current_song and self.songs:
            self.current_song = str(self.songs[0])
            self.play_song()
        elif pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            self.play_btn.configure(text="▶")
            self.is_playing = False
        else:
            pygame.mixer.music.unpause()
            self.play_btn.configure(text="⏸")
            self.is_playing = True
            
    def play_song(self):
        pygame.mixer.music.load(self.current_song)
        pygame.mixer.music.play()
        self.play_btn.configure(text="⏸")
        self.is_playing = True
        self.song_label.configure(
            text=Path(self.current_song).stem
        )
        
    def stop(self):
        pygame.mixer.music.stop()
        self.play_btn.configure(text="▶")
        self.is_playing = False
        self.progress['value'] = 0
        
    def next_song(self):
        if not self.current_song or not self.songs:
            return
        current_idx = self.songs.index(Path(self.current_song))
        next_idx = (current_idx + 1) % len(self.songs)
        self.current_song = str(self.songs[next_idx])
        self.play_song()
        
    def previous_song(self):
        if not self.current_song or not self.songs:
            return
        current_idx = self.songs.index(Path(self.current_song))
        prev_idx = (current_idx - 1) % len(self.songs)
        self.current_song = str(self.songs[prev_idx])
        self.play_song()
        
    def show_playlist(self):
        playlist = tk.Toplevel(self)
        playlist.title("Playlist")
        playlist.geometry("300x400")
        playlist.configure(bg=ViewerTheme.BG_COLOR)
        
        listbox = tk.Listbox(
            playlist,
            bg=ViewerTheme.BG_COLOR,
            fg=ViewerTheme.FG_COLOR,
            font=ViewerTheme.MAIN_FONT,
            selectmode=tk.SINGLE
        )
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for song in self.songs:
            listbox.insert(tk.END, song.stem)
            
        def play_selected(event):
            selection = listbox.curselection()
            if selection:
                self.current_song = str(self.songs[selection[0]])
                self.play_song()
                
        listbox.bind('<Double-Button-1>', play_selected)
        
    def update_time(self):
        if self.is_playing and self.current_song:
            audio = MP3(self.current_song)
            current_time = pygame.mixer.music.get_pos() / 1000
            total_time = audio.info.length
            
            # Update progress bar
            progress = (current_time / total_time) * 100
            self.progress['value'] = progress
            
            # Update time label
            current_mins = int(current_time // 60)
            current_secs = int(current_time % 60)
            total_mins = int(total_time // 60)
            total_secs = int(total_time % 60)
            
            self.time_label.configure(
                text=f"{current_mins:02d}:{current_secs:02d} / "
                     f"{total_mins:02d}:{total_secs:02d}"
            )
            
        self.after(1000, self.update_time)

class TextSearchViewer(tk.Toplevel):
    """Main viewer window"""
    def __init__(self, desktop: VaporwaveDesktop, search: SemanticSearch):
        super().__init__()
        
        self.desktop = desktop
        self.search = search
        self.title("﻿ＴＥＸＴ  ＳＥＡＲＣＨ")
        self.geometry("1400x800")
        self.attributes('-alpha', 0.95)
        
        # Make window draggable
        self.bind("<Button-1>", self.start_drag)
        self.bind("<B1-Motion>", self.drag)
        
        # Add music player button
        self.music_player = None
        music_btn = tk.Button(
            self,
            text="Music Player",
            command=self.toggle_music_player,
            width=15,
            **ViewerTheme.BUTTON_STYLE
        )
        music_btn.pack(pady=10)
        
        self.setup_ui()
    
    def start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        
    def drag(self, event):
        x = self.winfo_x() + event.x - self._drag_start_x
        y = self.winfo_y() + event.y - self._drag_start_y
        self.geometry(f"+{x}+{y}")
        
    def setup_ui(self):
        # Configure styles
        self.configure(bg=ViewerTheme.BG_COLOR)
        ttk.Style().configure(
            ".",
            background=ViewerTheme.BG_COLOR,
            foreground=ViewerTheme.FG_COLOR
        )
        
        # Main split pane
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Search and results
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=2)
        
        # Search toolbar
        self.search_bar = SearchToolbar(left_frame)
        self.search_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Results list
        self.results_list = ResultsList(left_frame)
        self.results_list.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Collections
        self.collections = CollectionPanel(paned)
        paned.add(self.collections, weight=1)
        
        # Connect events
        self.search_bar.on_search = self.perform_search
        self.results_list.on_view_context = self.view_context
        self.results_list.on_add_to_collection = self.add_to_collection
        
    def toggle_music_player(self):
        if self.music_player is None or not self.music_player.winfo_exists():
            self.music_player = RetroMusicPlayer(self)
        else:
            self.music_player.lift()
        
    def perform_search(self):
        """Execute search and display results"""
        query = self.search_bar.search_var.get()
        category = self.search_bar.category_var.get().lower()
        
        category_filter = category if category != "all" else None
        results = self.search.search(query, category_filter=category_filter)
        
        self.results_list.clear()
        for result in results:
            self.results_list.add_result(result)
            
    def view_context(self, result: SearchResult):
        """Show result in context window"""
        dialog = tk.Toplevel(self)
        dialog.title("View Context")
        dialog.geometry("800x600")
        
        text = tk.Text(
            dialog,
            wrap=tk.WORD,
            **ViewerTheme.TEXT_STYLE
        )
        text.pack(fill=tk.BOTH, expand=True)
        
        # Add context text
        header = " > ".join(result.header_path)
        text.insert(tk.END, f"=== {header} ===\n\n")
        text.insert(tk.END, result.text)
        
    def add_to_collection(self, result: SearchResult):
        """Add result to selected collection"""
        if not self.collections.collections:
            if messagebox.askyesno(
                "No Collections",
                "No collections exist. Create one now?"
            ):
                self.collections.new_collection()
            else:
                return
                
        collection = self.collections.notebook.select()
        if collection:
            name = self.collections.notebook.tab(collection, "text")
            self.collections.add_quote(name, result)

class MockSearch:
    """Mock search implementation for testing the viewer"""
    def __init__(self):
        self.content = [
            SearchResult(
                text="In a cyberpunk future where neon lights pierce through perpetual rain, "
                     "the boundaries between human consciousness and digital reality blur. "
                     "Mega-corporations tower over crowded streets, their holographic "
                     "advertisements painting the smog-filled sky.",
                similarity=0.95,
                header_path=["Chapter 1", "The Digital Age"],
                category="body",
                metadata={"page": 1, "position": 0}
            ),
            SearchResult(
                text="The rise of artificial intelligence brought both salvation and damnation. "
                     "As machines gained consciousness, humanity faced its greatest creation "
                     "and potentially its ultimate destruction.",
                similarity=0.87,
                header_path=["Chapter 2", "Machine Dreams"],
                category="body",
                metadata={"page": 2, "position": 0}
            ),
            SearchResult(
                text="Neural networks became the new nervous system of society, connecting "
                     "minds across the digital void. Information flowed like blood through "
                     "fiber optic veins, feeding the ever-hungry data markets.",
                similarity=0.82,
                header_path=["Chapter 2", "Digital Consciousness"],
                category="body",
                metadata={"page": 2, "position": 1}
            ),
            SearchResult(
                text="In the neon-lit underground, hackers traded secrets like currency. "
                     "Their world was a maze of code and neural pathways, where reality "
                     "itself could be rewritten with a few keystrokes.",
                similarity=0.78,
                header_path=["Chapter 3", "Underground Networks"],
                category="body",
                metadata={"page": 3, "position": 0}
            ),
            SearchResult(
                text="The city never sleeps, its dreams are encoded in binary.",
                similarity=0.92,
                header_path=["Chapter 3", "City Lights"],
                category="headers",
                metadata={"page": 3, "position": 1}
            )
        ]
    
    @classmethod
    def load(cls, path: Path) -> 'MockSearch':
        """Mock load method"""
        return cls()
    
    def search(self, query: str, category_filter: Optional[str] = None) -> List[SearchResult]:
        """Return mock search results"""
        results = self.content
        if category_filter:
            results = [r for r in results if r.category == category_filter]
        return results

def run_mock_viewer():
    """Run the viewer with mock data for testing"""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Create resources folders if they don't exist
    for folder in ['desktops', 'gifs', 'music']:
        path = Path('resources') / folder
        path.mkdir(parents=True, exist_ok=True)
    
    # Check if we have a background image, if not create a simple one
    bg_path = Path('resources/desktops/windows-xp.jpg')
    if not bg_path.exists():
        from PIL import Image
        img = Image.new('RGB', (1920, 1080), '#000066')
        img.save(bg_path)
    
    # Create desktop and viewer windows
    desktop = VaporwaveDesktop()
    viewer = TextSearchViewer(desktop, MockSearch())
    
    root.mainloop()

if __name__ == "__main__":
    run_mock_viewer()
