from __future__ import annotations
import tkinter as tk
from epubkit.debug import debug_test
from epubkit.dialogueUI import HTMLTextViewer, DialogStyle
import pytest
from unittest.mock import MagicMock
import customtkinter as ctk

class TestHTMLViewer:
    @pytest.fixture
    def setup_viewer(self):
        """Setup test environment"""
        root = tk.Tk()
        style = DialogStyle()
        return root, style

    def test_hover_enable_disable(self, setup_viewer):
        """Test hover functionality can be enabled/disabled"""
        root, style = setup_viewer
        
        # Test with hover enabled
        viewer_enabled = HTMLTextViewer(root, style, enable_hover=True)
        assert hasattr(viewer_enabled.text, '_bind')
        assert '<Motion>' in viewer_enabled.text.bind()
        assert '<Leave>' in viewer_enabled.text.bind()
        
        # Test with hover disabled
        viewer_disabled = HTMLTextViewer(root, style, enable_hover=False)
        assert '<Motion>' not in viewer_disabled.text.bind()
        assert '<Leave>' not in viewer_disabled.text.bind()

    def test_hover_highlighting(self, setup_viewer):
        """Test hover highlighting behavior"""
        root, style = setup_viewer
        viewer = HTMLTextViewer(root, style, enable_hover=True)
        
        # Insert test content with block tags
        viewer.text.insert("1.0", "Test Block")
        start, end = "1.0", "1.9"
        viewer.text.tag_add("block_0", start, end)
        
        # Simulate hover event
        event = MagicMock()
        event.x, event.y = 5, 5  # Coordinates within text
        viewer._on_hover(event)
        
        # Verify hover tag applied
        assert viewer.text.tag_ranges("hover")
        assert viewer.current_hover_tag == "block_0"
        
        # Simulate leave event
        viewer._on_leave(event)
        
        # Verify hover tag removed
        assert not viewer.text.tag_ranges("hover")
        assert viewer.current_hover_tag is None

    def test_html_formatting(self, setup_viewer):
        """Test HTML formatting and rendering"""
        root, style = setup_viewer
        viewer = HTMLTextViewer(root, style)
        
        test_html = """
        <p>Normal text</p>
        <p><b>Bold text</b></p>
        <p><i>Italic text</i></p>
        <p><u>Underlined text</u></p>
        """
        
        viewer.set_html_content(test_html)
        
        # Verify formatting tags applied
        assert "bold" in viewer.text.tag_names()
        assert "italic" in viewer.text.tag_names()
        assert "underline" in viewer.text.tag_names()

    def test_selection_callback(self, setup_viewer):
        """Test selection callback functionality"""
        root, style = setup_viewer
        
        # Create mock callback
        mock_callback = MagicMock()
        viewer = HTMLTextViewer(root, style, on_select=mock_callback)
        
        # Insert test content
        viewer.text.insert("1.0", "Test Block")
        start, end = "1.0", "1.9"
        block_tag = "block_0"
        viewer.text.tag_add(block_tag, start, end)
        viewer.block_map[block_tag] = {"test": "data"}
        
        # Simulate click event
        event = MagicMock()
        event.x, event.y = 5, 5
        viewer._on_click(event)
        
        # Verify callback called with correct data
        mock_callback.assert_called_once_with({"test": "data"})


class TestHTMLTextViewer:
    @pytest.fixture
    def setup_viewer(self):
        """Setup test environment"""
        root = tk.Tk()
        style = DialogStyle()
        viewer = HTMLTextViewer(root, style)
        return root, viewer

    def test_initial_content_loading(self, setup_viewer):
        """Test basic HTML content loading"""
        root, viewer = setup_viewer
        
        test_html = """
        <p>Normal text</p>
        <p><b>Bold text</b></p>
        <p><i>Italic text</i></p>
        """
        
        viewer.set_html_content(test_html)
        
        # Verify content loaded
        content = viewer.text.get("1.0", "end-1c")
        assert "Normal text" in content
        assert "Bold text" in content
        assert "Italic text" in content
        
        # Verify formatting tags
        bold_ranges = viewer.text.tag_ranges("bold")
        italic_ranges = viewer.text.tag_ranges("italic")
        assert bold_ranges  # Bold tag applied
        assert italic_ranges  # Italic tag applied

    def test_content_formatting(self, setup_viewer):
        """Test HTML formatting preservation"""
        root, viewer = setup_viewer
        
        test_html = """
        <p>Text with <b>bold</b> and <i>italic</i> and <u>underline</u></p>
        """
        
        viewer.set_html_content(test_html)
        
        # Check tag application
        for tag in ["bold", "italic", "underline"]:
            ranges = viewer.text.tag_ranges(tag)
            assert ranges, f"{tag} formatting not applied"

    def test_resize_formatting(self, setup_viewer):
        """Test formatting persistence after resize"""
        root, viewer = setup_viewer
        
        test_html = """
        <p>Text with <b>bold</b> and <i>italic</i> formatting</p>
        """
        
        viewer.set_html_content(test_html)
        
        # Get initial formatting
        initial_bold = viewer.text.tag_ranges("bold")
        initial_italic = viewer.text.tag_ranges("italic")
        
        # Simulate resize
        viewer._last_width = 100  # Force resize
        viewer._on_resize(None)
        root.update_idletasks()
        
        # Check formatting preserved
        assert viewer.text.tag_ranges("bold") == initial_bold
        assert viewer.text.tag_ranges("italic") == initial_italic

    def test_chunked_loading(self, setup_viewer):
        """Test chunked content loading"""
        root, viewer = setup_viewer
        viewer.chunk_size = 50  # Small chunks for testing
        
        # Create large content
        blocks = ["<p>Block {}</p>".format(i) for i in range(10)]
        test_html = "\n".join(blocks)
        
        viewer.set_html_content(test_html)
        
        # Verify chunks created
        assert len(viewer.content_chunks) > 1
        
        # Verify first chunk loaded
        assert "Block 0" in viewer.text.get("1.0", "end-1c")
        
        # Simulate scroll to load more
        viewer._on_scroll(type('Event', (), {'delta': -1}))
        root.update_idletasks()
        
        # Verify more content loaded
        full_content = viewer.text.get("1.0", "end-1c")
        assert "Block 1" in full_content

    def test_block_tracking(self, setup_viewer):
        """Test HTML block tracking"""
        root, viewer = setup_viewer
        
        test_html = """
        <p id="p1">First paragraph</p>
        <p id="p2">Second paragraph</p>
        """
        
        viewer.set_html_content(test_html)
        
        # Verify blocks tracked
        assert len(viewer.block_map) == 2
        
        # Verify block content
        for block in viewer.block_map.values():
            assert block.text_start is not None
            assert block.text_end is not None
            assert block.content in ["First paragraph", "Second paragraph"]
    
    @debug_test(['width', 'content', 'line_lengths'])
    def test_fixed_width_formatting(self, setup_viewer):
        """Test fixed-width text formatting"""
        root, viewer = setup_viewer
        root.geometry("100x100")
        root.update_idletasks()

        new_viewer = HTMLTextViewer(root, viewer.style, fixed_width=True)
        new_viewer.text.configure(width=20)
        new_viewer.pack(fill="both", expand=True)
      # Short width for testing
        
        long_text = "This is a very long paragraph that should be wrapped at word boundaries"
        test_html = f"<p>{long_text}</p>"
        
        new_viewer.set_html_content(test_html)
        # Verify text wrapped
        content = _track('content', new_viewer.text.get("1.0", "end-1c")) # type: ignore
        lines = content.split('\n')

        line_lengths = _track('line_lengths', [len(line.strip()) for line in lines]) # type: ignore

        assert all(len(line.strip()) <= 20 for line in lines if line.strip())

    def test_multiple_resizes(self, setup_viewer):
        """Test multiple resize operations"""
        root, viewer = setup_viewer
        
        test_html = "<p>Test content with <b>formatting</b></p>"
        viewer.set_html_content(test_html)
        
        # Multiple resizes
        widths = [100, 200, 150, 300]
        for width in widths:
            viewer._last_width = width - 1  # Force resize
            viewer.text.configure(width=width)
            viewer._on_resize(None)
            root.update_idletasks()
            
            # Verify content and formatting preserved
            assert "Test content" in viewer.text.get("1.0", "end-1c")
            assert viewer.text.tag_ranges("bold")

    @pytest.mark.manual
    def test_visual_formatting(self):
        """Manual test for visual verification"""
        root = tk.Tk()
        style = DialogStyle()
        viewer = HTMLTextViewer(root, style)
        viewer.pack(fill="both", expand=True)
        
        test_html = """
        <p>Normal paragraph</p>
        <p><b>Bold paragraph</b></p>
        <p><i>Italic paragraph</i></p>
        <p>Mixed <b>bold</b> and <i>italic</i> text</p>
        """
        
        viewer.set_html_content(test_html)
        root.mainloop()